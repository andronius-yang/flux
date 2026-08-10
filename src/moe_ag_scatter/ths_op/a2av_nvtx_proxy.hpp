//===- a2av_nvtx_proxy.hpp ------------------------------------ C++ ------===//
//
// Copyright 2025 ByteDance Ltd. and/or its affiliates. All rights reserved.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
//===----------------------------------------------------------------------===//
// FLUX_A2AV_NVTX_PROXY=1: device code cannot emit NVTX, so this poller thread
// proxies it. The a2av GEMM publishes per-source-bucket progress into
// host-mapped A2AVProgressSlots (a2av_progress.h); this thread polls at
// ~10 us cadence and emits live NVTX ranges on the "a2av" domain, so the
// per-source structure of the otherwise-opaque grouped-GEMM span shows up on
// an nsys timeline (trace set cuda,nvtx — no extra flags needed).
//
// Per source s and iteration (epoch e = NVSHMEM run-id):
//   i<e>.src<s>.wait     iteration start -> source payload observed arrived
//   i<e>.src<s>.pending  arrived -> first tile of s fired (claimer saturated)
//   i<e>.src<s>.compute  first tile fired -> all of s's tiles retired
//   i<e>.src<s>.c0-25 .. c75-100   completion-quantile sub-ranges (straggler
//                        tail visible at a glance); needs expected[] totals
// plus aggregate envelopes i<e>.intra_epoch / i<e>.inter_epoch (first fire ->
// all drained over the local-node vs remote-node source sets) and an i<e>
// bracket range. Emission uses nvtxDomainRangeStartEx/End correlation-ID
// ranges — never push/pop, which is a per-thread stack and would mis-nest the
// freely overlapping per-source ranges.
//
// Accuracy: every edge is an observation, delayed by poll cadence + PCIe
// write visibility (~5-30 us worst case). Ordering between edges seen in the
// same tick is not meaningful. This is a timeline aid, not a metrics source —
// quote latencies from e2e-mode sweep cells only (sweeps/SCHEMA.md).
//
// The layer-C SIDECAR, unlike the live view, is lossless by construction:
// forward enqueues, in stream order right after the GEMM, a D2H snapshot of
// the device slots into a pinned per-epoch ring plus a host callback that
// publishes it (enqueue_epoch_snapshot). The snapshot is the epoch's exact
// final state — arrival stamps, t0, end trace cursor — regardless of when
// any host thread gets scheduled; the poller merely drains the ring. The old
// design sampled the live slots through a single latest-value mailbox, so an
// epoch whose start AND end both fell inside one poller stall (NFS flush,
// CE-starved refresh, OS deschedule, pause window) vanished or merged into a
// neighbor block. Ring depth bounds only drain lag; overflow drops oldest
// epochs EXPLICITLY (stderr), never silently.

#pragma once

#include <cuda_runtime_api.h>
#include <nvtx3/nvToolsExt.h>

#include <atomic>
#include <chrono>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iterator>
#include <map>
#include <mutex>
#include <thread>
#include <utility>
#include <vector>

#include "flux/a2av_progress.h"

namespace bytedance::flux {

class A2AVNvtxProxy {
 public:
  // slots: pinned host copy (read here); dev_slots: kernel-written device
  // copy, refreshed into `slots` each tick with a small CE memcpy on
  // `stream`. CE, not an SM helper: under CUDA_DEVICE_MAX_CONNECTIONS=1 a
  // resident mirror kernel blocks the compute queue and deadlocks the GEMM,
  // while CE copies interleave with a running kernel (same property the hier
  // wire itself relies on).
  A2AVNvtxProxy(
      A2AVProgressSlots *slots,
      const A2AVProgressSlots *dev_slots,
      cudaStream_t stream,
      int rank,
      int world_size,
      int nnodes)
      : slots_(slots),
        dev_slots_(dev_slots),
        stream_(stream),
        rank_(rank),
        world_size_(world_size),
        nb_(world_size + 1),
        local_world_(nnodes > 0 ? world_size / nnodes : world_size),
        my_node_(local_world_ > 0 ? rank / local_world_ : 0) {
    // one domain per source: nsys renders each domain as its own track, so a
    // source's sequential wait/pending/compute ranges occupy a single row
    // (quartiles nest on one sub-row) instead of all sources stacking into an
    // arbitrary pile on one track. Zero-padded names keep tracks sorted.
    domains_.resize(nb_);
    for (int s = 0; s < nb_; ++s) {
      char nm[24];
      if (s == world_size_) {
        snprintf(nm, sizeof(nm), "a2av/multi");
      } else {
        snprintf(nm, sizeof(nm), "a2av/src%02d", s);
      }
      domains_[s] = nvtxDomainCreateA(nm);
    }
    dom_epoch_ = nvtxDomainCreateA("a2av/epochs");
    src_.resize(nb_);
    if (cudaHostAlloc((void **)&snap_ring_, sizeof(EpochSnap) * kSnapRing, cudaHostAllocDefault) !=
        cudaSuccess) {
      snap_ring_ = nullptr;  // sidecar disabled; live NVTX view still works
      fprintf(
          stderr,
          "[a2av-nvtx-proxy] rank %d: snapshot ring alloc failed, "
          "tile-trace sidecar disabled\n",
          rank);
    }
    thread_ = std::thread([this] { run(); });
  }

  ~A2AVNvtxProxy() {
    stop_.store(true, std::memory_order_release);
    if (thread_.joinable()) {
      thread_.join();
    }
    if (sidecar_ != nullptr) {
      fclose(sidecar_);
    }
    if (trace_staging_ != nullptr) {
      cudaFreeHost(trace_staging_);
    }
    if (snap_ring_ != nullptr) {
      cudaFreeHost(snap_ring_);
    }
    // domains intentionally leaked: nvtxDomainDestroy may race tool teardown
  }

  // layer C: kernel-written per-tile trace buffer (device). Called once from
  // the op's lazy alloc, strictly before that iteration's iter_start_cb —
  // visibility to the poller rides the epoch release/acquire pair.
  void
  set_tile_trace(const A2AVTileRecord *dev, uint32_t capacity) {
    trace_dev_ = dev;
    trace_capacity_ = capacity;
  }

  // Pause the poller's CUDA activity while the main thread issues NVSHMEM
  // on-stream ops (FLUX_A2AV_EARLY_LAUNCH's deferred wire): concurrent
  // cudaStreamSynchronize here vs NVSHMEM host calls there deadlocks
  // (observed as a first-forward hang; lock-order inversion between the
  // NVSHMEM host lib and the driver). Costs a few observation ticks.
  void
  set_paused(bool p) {
    paused_.store(p, std::memory_order_release);
  }

  A2AVNvtxProxy(const A2AVNvtxProxy &) = delete;
  A2AVNvtxProxy &operator=(const A2AVNvtxProxy &) = delete;

  // pinned per-epoch snapshot ring: depth bounds only how far the sidecar
  // drain may lag the stream (deepest realistic profile run is ~15 epochs;
  // 32 lets the drain sleep for an entire run and still lose nothing)
  static constexpr uint32_t kSnapRing = 32;
  static constexpr uint32_t kNoSlot = 0xFFFFFFFFu;
  struct EpochSnap {
    A2AVProgressSlots slots;
    uint64_t epoch;
  };

  // ---- stream-ordered callbacks (cudaLaunchHostFunc payloads) -------------
  // iter_start is enqueued immediately before the GEMM launch, iter_end
  // immediately after: they execute at those points in stream order, so they
  // bracket the kernel without any device sync. Callbacks only touch atomics
  // (no CUDA API calls — forbidden in host funcs).
  struct IterStart {
    A2AVNvtxProxy *self;
    uint64_t epoch;
    // dense-schedule per-source expected tile totals (host-computed); empty =
    // unknown (claimer mode publishes totals via slots->expected instead)
    std::vector<uint32_t> expected;
    // per-source ROW counts (sum over experts of the logical splits): the
    // ground truth for "does source s contribute data" — tile attribution
    // (expected) cannot distinguish a small segment from an empty one
    std::vector<uint32_t> rows;
  };
  static void CUDART_CB
  iter_start_cb(void *p) {
    IterStart *m = static_cast<IterStart *>(p);
    {
      std::lock_guard<std::mutex> g(m->self->mu_);
      m->self->pending_meta_[m->epoch] = {std::move(m->expected), std::move(m->rows)};
      // paranoia bound; the drain erases consumed entries
      while (m->self->pending_meta_.size() > 4 * kSnapRing) {
        m->self->pending_meta_.erase(m->self->pending_meta_.begin());
      }
    }
    m->self->iter_epoch_.store(m->epoch, std::memory_order_release);
    delete m;
  }

  // -- lossless per-epoch sidecar capture -----------------------------------
  // Called by forward at the old iter_end position (after the GEMM launch in
  // stream order; after the deferred wire in HOST order — a pending host
  // function blocks the single channel under CUDA_DEVICE_MAX_CONNECTIONS=1).
  // The memcpy snapshots the epoch's final device slots into ring slot k;
  // the callback then publishes it and marks the epoch ended for the live
  // NVTX view. Single producer (the forward thread).
  void
  enqueue_epoch_snapshot(cudaStream_t stream, uint64_t epoch) {
    uint32_t slot = kNoSlot;
    if (snap_ring_ != nullptr) {
      slot = (uint32_t)(snap_enqueued_ % kSnapRing);
      cudaError_t rc = cudaMemcpyAsync(
          &snap_ring_[slot].slots,
          dev_slots_,
          sizeof(A2AVProgressSlots),
          cudaMemcpyDeviceToHost,
          stream);
      if (rc != cudaSuccess) {
        if (!snap_warned_) {
          snap_warned_ = true;
          fprintf(
              stderr,
              "[a2av-nvtx-proxy] rank %d: epoch snapshot enqueue failed: %s\n",
              rank_,
              cudaGetErrorString(rc));
        }
        slot = kNoSlot;
      } else {
        snap_enqueued_ += 1;
      }
    }
    SnapMark *m = new SnapMark{this, epoch, slot};
    if (cudaLaunchHostFunc(stream, snap_cb, m) != cudaSuccess) {
      // without the callback the slot is never published; the epoch is lost
      // for the sidecar but the run must not die
      snap_enqueued_ -= (slot != kNoSlot) ? 1 : 0;
      delete m;
    }
  }

  struct SnapMark {
    A2AVNvtxProxy *self;
    uint64_t epoch;
    uint32_t slot;  // kNoSlot: publish nothing, only mark the epoch ended
  };
  static void CUDART_CB
  snap_cb(void *p) {
    SnapMark *m = static_cast<SnapMark *>(p);
    if (m->slot != kNoSlot) {
      m->self->snap_ring_[m->slot].epoch = m->epoch;  // memcpy done: stream order
      m->self->snap_produced_.fetch_add(1, std::memory_order_release);
    }
    m->self->end_epoch_.store(m->epoch, std::memory_order_release);
    delete m;
  }

 private:
  enum class Phase : uint8_t { kWait, kPending, kCompute, kDone };

  struct SrcState {
    Phase phase = Phase::kDone;
    uint32_t base_claimed = 0;
    uint32_t base_completed = 0;
    uint32_t last_completed = 0;
    uint32_t expected = 0;
    bool has_expected = false;
    int qidx = -1;  // open completion-quantile sub-range, -1 = none
    nvtxRangeId_t r_main = 0;
    nvtxRangeId_t r_q = 0;
  };

  // ARGB range colors: wire-latency wait vs saturation pending vs compute,
  // with inter-node sources tinted red and intra-node green
  static constexpr uint32_t kColWait = 0xFF9E9E9E;
  static constexpr uint32_t kColPending = 0xFFF9A825;
  static constexpr uint32_t kColIntra = 0xFF2E7D32;
  static constexpr uint32_t kColInter = 0xFFC62828;
  static constexpr uint32_t kColEpoch = 0xFF1565C0;
  static constexpr uint32_t kColIter = 0xFF546E7A;

  bool
  is_inter(int s) const {
    return s < world_size_ && (s / local_world_) != my_node_;
  }

  uint32_t
  volatile_u32(const uint32_t *p) const {
    return *reinterpret_cast<const volatile uint32_t *>(p);
  }
  uint64_t
  volatile_u64(const uint64_t *p) const {
    return *reinterpret_cast<const volatile uint64_t *>(p);
  }

  nvtxRangeId_t
  range_start(nvtxDomainHandle_t dom, uint32_t color, const char *fmt, ...) {
    char buf[96];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(buf, sizeof(buf), fmt, ap);
    va_end(ap);
    nvtxEventAttributes_t a = {};
    a.version = NVTX_VERSION;
    a.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
    a.colorType = NVTX_COLOR_ARGB;
    a.color = color;
    a.messageType = NVTX_MESSAGE_TYPE_ASCII;
    a.message.ascii = buf;
    return nvtxDomainRangeStartEx(dom, &a);
  }

  void
  mark_progress(int s, uint32_t claimed, uint32_t completed) {
    char buf[96];
    snprintf(
        buf,
        sizeof(buf),
        "i%llu.%s %u/%u",
        (unsigned long long)cur_epoch_,
        src_name(s),
        claimed,
        completed);
    nvtxEventAttributes_t a = {};
    a.version = NVTX_VERSION;
    a.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
    a.messageType = NVTX_MESSAGE_TYPE_ASCII;
    a.message.ascii = buf;
    a.payloadType = NVTX_PAYLOAD_TYPE_UNSIGNED_INT64;
    a.payload.ullValue = (uint64_t(claimed) << 32) | completed;
    nvtxDomainMarkEx(domains_[s], &a);
  }

  // device-precise arrival: NVTX range edges are live observations and cannot
  // be backdated, so the exact %globaltimer arrival rides as a mark payload
  // (ns since t0_gt; 0 = stamp not available, e.g. never visibly blocked)
  void
  mark_arrival(int s) {
    if (s >= world_size_ || volatile_u64(&slots_->ready_seq[s]) < cur_epoch_) {
      return;  // no device arrival stamp for this source this epoch
    }
    uint64_t rel = 0;
    if (volatile_u64(&slots_->t0_seq) >= cur_epoch_) {
      rel = volatile_u64(&slots_->arrival_gt[s]) - volatile_u64(&slots_->t0_gt);
    }
    char buf[64];
    snprintf(buf, sizeof(buf), "i%llu.%s.arrival", (unsigned long long)cur_epoch_, src_name(s));
    nvtxEventAttributes_t a = {};
    a.version = NVTX_VERSION;
    a.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
    a.messageType = NVTX_MESSAGE_TYPE_ASCII;
    a.message.ascii = buf;
    a.payloadType = NVTX_PAYLOAD_TYPE_UNSIGNED_INT64;
    a.payload.ullValue = rel;
    nvtxDomainMarkEx(domains_[s], &a);
  }

  // layer C sidecar: one self-contained block per snapshot (= per iteration)
  // under FLUX_SWEEP_RECORD_DIR (the sweep runner exports it per cell; unset
  // on ad-hoc runs -> skip with one stderr note). Sourced entirely from the
  // stream-ordered snapshot, never the live slots.
  void
  dump_epoch(const EpochSnap &es) {
    uint32_t cur = es.slots.trace_cursor;
    uint32_t n = snap_base_valid_ ? cur - snap_trace_base_ : 0;  // resync: no span
    uint32_t base = cur - n;
    snap_trace_base_ = cur;
    snap_base_valid_ = true;
    if (trace_dev_ == nullptr) {
      return;
    }
    if (n > trace_capacity_) {
      n = trace_capacity_;  // cursor past capacity = dropped records
      base = cur - n;
    }
    const char *dir = getenv("FLUX_SWEEP_RECORD_DIR");
    if (dir == nullptr || dir[0] == '\0') {
      if (!sidecar_warned_) {
        sidecar_warned_ = true;
        fprintf(
            stderr,
            "[a2av-nvtx-proxy] rank %d: FLUX_SWEEP_RECORD_DIR unset, tile trace not saved\n",
            rank_);
      }
      return;
    }
    if (trace_staging_ == nullptr) {
      if (cudaHostAlloc(
              (void **)&trace_staging_,
              sizeof(A2AVTileRecord) * trace_capacity_,
              cudaHostAllocDefault) != cudaSuccess) {
        return;
      }
    }
    // ring over the monotonic cursor: this epoch's records occupy
    // [base % cap, base % cap + n) mod cap — up to two linear spans
    uint32_t off = base % trace_capacity_;
    uint32_t first = n <= trace_capacity_ - off ? n : trace_capacity_ - off;
    cudaError_t rc = cudaSuccess;
    if (n > 0) {
      rc = cudaMemcpyAsync(
          trace_staging_,
          trace_dev_ + off,
          sizeof(A2AVTileRecord) * first,
          cudaMemcpyDeviceToHost,
          stream_);
      if (rc == cudaSuccess && n > first) {
        rc = cudaMemcpyAsync(
            trace_staging_ + first,
            trace_dev_,
            sizeof(A2AVTileRecord) * (n - first),
            cudaMemcpyDeviceToHost,
            stream_);
      }
      if (rc == cudaSuccess) {
        rc = cudaStreamSynchronize(stream_);
      }
      if (rc != cudaSuccess) {
        return;
      }
      // torn-span check: a still-running later epoch may have lapped the ring
      // while we copied. The live cursor after the copy bounds its writes.
      refresh();
      uint32_t live = volatile_u32(&slots_->trace_cursor);
      if (live - base > trace_capacity_) {
        fprintf(
            stderr,
            "[a2av-nvtx-proxy] rank %d i%llu: trace ring lapped during drain "
            "(%u records possibly torn); raise FLUX_A2AV_TRACE_EPOCHS\n",
            rank_,
            (unsigned long long)es.epoch,
            live - base - trace_capacity_);
      }
    }
    if (sidecar_ == nullptr) {
      char path[512];
      snprintf(path, sizeof(path), "%s/a2av_tile_trace_r%d.bin", dir, rank_);
      sidecar_ = fopen(path, "wb");
      if (sidecar_ == nullptr) {
        return;
      }
    }
    // expected: host-computed dense-schedule totals when the meta path ran,
    // else the kernel-published claimer totals (epoch-tag gated); rows: host
    // meta only. Both keyed by epoch — a late drain still gets its own meta.
    std::vector<uint32_t> exp(nb_, 0u), rows(nb_, 0u);
    bool have_host_exp = false;
    {
      std::lock_guard<std::mutex> g(mu_);
      auto it = pending_meta_.find(es.epoch);
      if (it != pending_meta_.end()) {
        for (int s = 0; s < nb_ && s < (int)it->second.first.size(); ++s) {
          exp[s] = it->second.first[s];
        }
        have_host_exp = !it->second.first.empty();
        for (int s = 0; s < nb_ && s < (int)it->second.second.size(); ++s) {
          rows[s] = it->second.second[s];
        }
        pending_meta_.erase(pending_meta_.begin(), std::next(it));
      }
    }
    if (!have_host_exp && es.slots.expected_seq >= es.epoch) {
      for (int s = 0; s < nb_; ++s) {
        exp[s] = es.slots.expected[s];
      }
    }
    struct Header {
      uint32_t magic, version;
      uint64_t epoch;
      int32_t rank, world_size, nnodes, nb;
      uint64_t t0_gt;
      uint32_t n_records, pad;
    } h = {
        0xa2a71e5u,
        2u,
        es.epoch,
        rank_,
        world_size_,
        local_world_ > 0 ? world_size_ / local_world_ : 1,
        nb_,
        es.slots.t0_gt,
        n,
        0u};
    fwrite(&h, sizeof(h), 1, sidecar_);
    fwrite((const void *)es.slots.arrival_gt, sizeof(uint64_t), nb_, sidecar_);
    // ready_seq lets the reader validate arrival_gt (stamp is per-epoch valid
    // iff ready_seq[s] >= epoch; otherwise it is stale or never written)
    fwrite((const void *)es.slots.ready_seq, sizeof(uint64_t), nb_, sidecar_);
    fwrite(exp.data(), sizeof(uint32_t), nb_, sidecar_);
    fwrite(rows.data(), sizeof(uint32_t), nb_, sidecar_);
    fwrite(trace_staging_, sizeof(A2AVTileRecord), n, sidecar_);
    fflush(sidecar_);
  }

  // consume ready snapshots in order (poller thread; also the teardown
  // path). Timeliness affects only device trace-ring headroom, never whether
  // an epoch's block is written.
  void
  drain_snapshots() {
    if (snap_ring_ == nullptr) {
      return;
    }
    uint64_t prod = snap_produced_.load(std::memory_order_acquire);
    if (prod - snap_consumed_ > kSnapRing) {
      uint64_t drop = prod - kSnapRing - snap_consumed_;
      fprintf(
          stderr,
          "[a2av-nvtx-proxy] rank %d: snapshot ring overflow, %llu epoch(s) dropped\n",
          rank_,
          (unsigned long long)drop);
      snap_consumed_ = prod - kSnapRing;
      snap_base_valid_ = false;  // record spans across the gap are unknowable
    }
    while (snap_consumed_ < prod) {
      dump_epoch(snap_ring_[snap_consumed_ % kSnapRing]);
      snap_consumed_ += 1;
    }
  }

  const char *
  src_name(int s) {
    if (s == world_size_) {
      return "multi";
    }
    snprintf(name_buf_, sizeof(name_buf_), "src%d", s);
    return name_buf_;
  }

  // refresh the pinned copy from device (~800 B CE copy; the sync doubles as
  // poll pacing). Failure is terminal for profiling but must not kill the run.
  void
  refresh() {
    cudaError_t rc = cudaMemcpyAsync(
        slots_, dev_slots_, sizeof(A2AVProgressSlots), cudaMemcpyDeviceToHost, stream_);
    if (rc == cudaSuccess) {
      rc = cudaStreamSynchronize(stream_);
    }
    if (rc != cudaSuccess && !copy_failed_) {
      copy_failed_ = true;
      fprintf(
          stderr, "[a2av-nvtx-proxy] rank %d refresh failed: %s\n", rank_, cudaGetErrorString(rc));
    }
  }

  // close out the active epoch's LIVE NVTX state: if its end callback already
  // fired, take a final observation so ranges end at their true state. Called
  // from the normal end path, from a late-noticed epoch switch (back-to-back
  // iterations can outrun the poll cadence, especially under nsys/CUPTI
  // overhead), and from thread exit. The sidecar no longer rides this path —
  // drain_snapshots() owns it and cannot miss epochs.
  void
  close_current(bool ended) {
    if (cur_epoch_ == 0) {
      return;
    }
    if (ended) {
      refresh();
      poll_once();
    }
    finish_epoch();
  }

  void
  run() {
    while (!stop_.load(std::memory_order_acquire)) {
      if (paused_.load(std::memory_order_acquire)) {
        relax();
        continue;
      }
      drain_snapshots();
      uint64_t e = iter_epoch_.load(std::memory_order_acquire);
      if (e != last_begun_) {
        close_current(end_epoch_.load(std::memory_order_acquire) >= cur_epoch_);
        refresh();  // baselines must reflect the previous iteration's final state
        begin_epoch(e);
      }
      if (cur_epoch_ != 0) {
        refresh();
        poll_once();
        if (end_epoch_.load(std::memory_order_acquire) >= cur_epoch_) {
          close_current(true);
        }
      }
      relax();
    }
    close_current(end_epoch_.load(std::memory_order_acquire) >= cur_epoch_);
    // teardown drain: the forward thread is quiesced (op dtor joins us after
    // the last iteration completed), so pause no longer matters — every
    // published snapshot gets its sidecar block even if we slept all run.
    drain_snapshots();
  }

  void
  begin_epoch(uint64_t e) {
    if (cur_epoch_ != 0) {
      finish_epoch();
    }
    last_begun_ = e;
    cur_epoch_ = e;
    if (e == 0) {
      return;
    }
    std::vector<uint32_t> expected;
    {
      // copy (not consume): dump_epoch owns the map's lifetime so a late
      // drain still finds its epoch's meta
      std::lock_guard<std::mutex> g(mu_);
      auto it = pending_meta_.find(e);
      if (it != pending_meta_.end()) {
        expected = it->second.first;
      }
    }
    r_iter_ = range_start(dom_epoch_, kColIter, "i%llu", (unsigned long long)e);
    expected_seq_ticks_ = 0;
    intra_open_ = inter_open_ = false;
    intra_live_ = inter_live_ = 0;
    for (int s = 0; s < nb_; ++s) {
      SrcState &st = src_[s];
      st.base_claimed = volatile_u32(&slots_->claimed[s]);
      st.base_completed = volatile_u32(&slots_->completed[s]);
      st.last_completed = 0;
      st.qidx = -1;
      st.has_expected = s < (int)expected.size();
      st.expected = st.has_expected ? expected[s] : 0;
      st.phase = Phase::kWait;
      st.r_main =
          range_start(domains_[s], kColWait, "i%llu.%s.wait", (unsigned long long)e, src_name(s));
      if (s < world_size_) {
        (is_inter(s) ? inter_live_ : intra_live_) += 1;
      }
    }
  }

  void
  enter_compute(int s) {
    SrcState &st = src_[s];
    nvtxDomainRangeEnd(domains_[s], st.r_main);
    uint32_t col = is_inter(s) ? kColInter : kColIntra;
    st.r_main = range_start(
        domains_[s], col, "i%llu.%s.compute", (unsigned long long)cur_epoch_, src_name(s));
    st.phase = Phase::kCompute;
    st.qidx = 0;
    st.r_q = range_start(
        domains_[s], col, "i%llu.%s.c0-25", (unsigned long long)cur_epoch_, src_name(s));
    if (s < world_size_) {
      bool inter = is_inter(s);
      bool &open = inter ? inter_open_ : intra_open_;
      if (!open) {
        open = true;
        nvtxRangeId_t &r = inter ? r_inter_ : r_intra_;
        r = range_start(
            dom_epoch_,
            kColEpoch,
            "i%llu.%s_epoch",
            (unsigned long long)cur_epoch_,
            inter ? "inter" : "intra");
      }
    }
  }

  void
  finish_src(int s) {
    SrcState &st = src_[s];
    if (st.qidx >= 0) {
      nvtxDomainRangeEnd(domains_[s], st.r_q);
      st.qidx = -1;
    }
    if (st.phase != Phase::kDone) {
      nvtxDomainRangeEnd(domains_[s], st.r_main);
      st.phase = Phase::kDone;
      if (s < world_size_) {
        int &live = is_inter(s) ? inter_live_ : intra_live_;
        bool &open = is_inter(s) ? inter_open_ : intra_open_;
        live -= 1;
        if (live == 0 && open) {
          nvtxDomainRangeEnd(dom_epoch_, is_inter(s) ? r_inter_ : r_intra_);
          open = false;
        }
      }
    }
  }

  void
  poll_once() {
    if (volatile_u64(&slots_->expected_seq) >= cur_epoch_) {
      expected_seq_ticks_ += 1;
    }
    for (int s = 0; s < nb_; ++s) {
      SrcState &st = src_[s];
      if (st.phase == Phase::kDone) {
        continue;
      }
      uint32_t c = volatile_u32(&slots_->claimed[s]) - st.base_claimed;
      uint32_t d = volatile_u32(&slots_->completed[s]) - st.base_completed;
      // claimer mode publishes totals from the kernel; adopt them only after
      // the epoch tag survived two refreshes (one CE pass can pair a fresh
      // tag with stale totals)
      if (!st.has_expected && expected_seq_ticks_ >= 2) {
        st.expected = volatile_u32(&slots_->expected[s]);
        st.has_expected = true;
      }
      if (st.phase == Phase::kWait) {
        if (c > 0) {
          mark_arrival(s);
          enter_compute(s);  // fired before we saw ready (or multi bucket)
        } else if (s < world_size_ && volatile_u64(&slots_->ready_seq[s]) >= cur_epoch_) {
          mark_arrival(s);
          nvtxDomainRangeEnd(domains_[s], st.r_main);
          st.r_main = range_start(
              domains_[s],
              kColPending,
              "i%llu.%s.pending",
              (unsigned long long)cur_epoch_,
              src_name(s));
          st.phase = Phase::kPending;
        } else if (st.has_expected && st.expected == 0) {
          finish_src(s);  // nothing routed from this source this iteration
        }
      }
      if (st.phase == Phase::kPending && c > 0) {
        enter_compute(s);
      }
      if (st.phase == Phase::kCompute) {
        if (d != st.last_completed) {
          st.last_completed = d;
          mark_progress(s, c, d);
        }
        if (st.has_expected && st.expected > 0) {
          while (st.qidx >= 0 && st.qidx < 3 &&
                 4ull * d >= (unsigned)(st.qidx + 1) * (unsigned long long)st.expected) {
            nvtxDomainRangeEnd(domains_[s], st.r_q);
            st.qidx += 1;
            static const char *kQ[] = {"c0-25", "c25-50", "c50-75", "c75-100"};
            st.r_q = range_start(
                domains_[s],
                is_inter(s) ? kColInter : kColIntra,
                "i%llu.%s.%s",
                (unsigned long long)cur_epoch_,
                src_name(s),
                kQ[st.qidx]);
          }
          if (d >= st.expected) {
            finish_src(s);
          }
        }
      }
    }
  }

  void
  finish_epoch() {
    for (int s = 0; s < nb_; ++s) {
      finish_src(s);
    }
    if (intra_open_) {
      nvtxDomainRangeEnd(dom_epoch_, r_intra_);
      intra_open_ = false;
    }
    if (inter_open_) {
      nvtxDomainRangeEnd(dom_epoch_, r_inter_);
      inter_open_ = false;
    }
    nvtxDomainRangeEnd(dom_epoch_, r_iter_);
    cur_epoch_ = 0;
  }

  void
  relax() {
    auto t0 = std::chrono::steady_clock::now();
    while (std::chrono::steady_clock::now() - t0 < std::chrono::microseconds(10)) {
#if defined(__x86_64__)
      __builtin_ia32_pause();
#endif
    }
  }

  A2AVProgressSlots *slots_;
  const A2AVProgressSlots *dev_slots_;
  cudaStream_t stream_;
  bool copy_failed_ = false;
  const int rank_;
  const int world_size_;
  const int nb_;
  const int local_world_;
  const int my_node_;
  std::vector<nvtxDomainHandle_t> domains_;  // per source bucket, "a2av/srcNN"
  nvtxDomainHandle_t dom_epoch_ = nullptr;   // "a2av/epochs": iN + intra/inter

  std::atomic<bool> stop_{false};
  std::atomic<bool> paused_{false};
  std::atomic<uint64_t> iter_epoch_{0};
  std::atomic<uint64_t> end_epoch_{0};
  std::mutex mu_;
  // epoch -> (expected tiles, rows) from iter_start_cb; erased by dump_epoch
  std::map<uint64_t, std::pair<std::vector<uint32_t>, std::vector<uint32_t>>> pending_meta_;

  // per-epoch snapshot ring (pinned): forward-thread producer via
  // enqueue_epoch_snapshot, poller-thread consumer via drain_snapshots
  EpochSnap *snap_ring_ = nullptr;
  uint64_t snap_enqueued_ = 0;              // forward thread only
  std::atomic<uint64_t> snap_produced_{0};  // bumped by snap_cb after memcpy
  uint64_t snap_consumed_ = 0;              // poller thread only
  uint32_t snap_trace_base_ = 0;            // trace cursor at last drained epoch
  bool snap_base_valid_ = true;             // false across an overflow gap
  bool snap_warned_ = false;

  // poller-thread state
  uint64_t cur_epoch_ = 0;
  uint64_t last_begun_ = 0;
  int expected_seq_ticks_ = 0;
  // layer C tile-trace extraction
  const A2AVTileRecord *trace_dev_ = nullptr;
  uint32_t trace_capacity_ = 0;
  A2AVTileRecord *trace_staging_ = nullptr;
  FILE *sidecar_ = nullptr;
  bool sidecar_warned_ = false;
  std::vector<SrcState> src_;
  nvtxRangeId_t r_iter_ = 0, r_intra_ = 0, r_inter_ = 0;
  bool intra_open_ = false, inter_open_ = false;
  int intra_live_ = 0, inter_live_ = 0;
  char name_buf_[16];

  std::thread thread_;
};

}  // namespace bytedance::flux
