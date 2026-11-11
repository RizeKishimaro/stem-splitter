/* SonicLab native engine: C ABI prototype (Phase 3).
 *
 * A plain C interface so any language can drive the engine. Python's ctypes
 * is the first client, and nanobind can wrap the same functions later.
 * The prototype only implements the master stage of the Python engine:
 * output gain (clamped to 0..1.5), master mute, clip to ±1, and peak
 * metering. It does not mix clips yet.
 *
 * Threads
 *   control   Any thread except the audio thread. Calls create/destroy,
 *             the setters, start/stop, read_output, take_peak and get_stats.
 *             Setters are queued and applied at the start of the next block.
 *   audio     The engine's own device thread (start_device), or the caller
 *             of sl_engine_process. Never both at once. It never locks,
 *             allocates, or calls back into the client.
 *   There is deliberately no way to register a callback. Client code, and
 *   so the Python interpreter and its GIL, can never run on the audio thread.
 *
 * Memory ownership
 *   - sl_engine_create returns a handle the caller owns. Release it exactly
 *     once with sl_engine_destroy, which first stops the device thread.
 *   - Buffer pointers passed in are borrowed for the duration of the call
 *     only. The engine never keeps them and never frees them.
 *   - The engine never returns pointers to its own memory. Data comes back
 *     by copy, into caller buffers or caller structs.
 *   - Audio is interleaved float32, `frames * channels` values.
 *
 * Errors
 *   Functions that can fail return an int32_t status: SL_OK or a negative SL_ERR_*.
 */
#ifndef SONICLAB_ENGINE_CAPI_H
#define SONICLAB_ENGINE_CAPI_H

#include <stdint.h>

#if defined(_WIN32)
#  if defined(SONICLAB_ENGINE_BUILD)
#    define SL_API __declspec(dllexport)
#  else
#    define SL_API __declspec(dllimport)
#  endif
#else
#  define SL_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define SL_ENGINE_ABI_VERSION 1u

enum {
    SL_OK = 0,
    SL_ERR_NULL = -1,     /* null handle or buffer */
    SL_ERR_ARG = -2,      /* argument out of range */
    SL_ERR_BUSY = -3,     /* the device thread owns the audio side */
    SL_ERR_FULL = -4,     /* command queue full; retry later */
    SL_ERR_STATE = -5     /* e.g. stopping a device that is not running */
};

typedef struct sl_engine sl_engine;

/* Counters copied out by sl_engine_get_stats. Fixed-width fields only, so
 * the layout is identical for every compiler and for ctypes. */
typedef struct sl_engine_stats {
    int64_t blocks_processed;
    int64_t frames_processed;
    int64_t commands_applied;
    int64_t output_frames_dropped; /* the reader of read_output fell behind */
    int64_t max_block_nanoseconds; /* worst time spent processing one block */
} sl_engine_stats;

SL_API uint32_t sl_engine_abi_version(void);

/* NULL when the arguments are out of range or allocation fails.
 * channels 1..8, max_block_frames 1..8192, sample_rate 8000..384000. */
SL_API sl_engine* sl_engine_create(double sample_rate, int32_t channels, int32_t max_block_frames);
SL_API void sl_engine_destroy(sl_engine* engine);

SL_API int32_t sl_engine_channels(const sl_engine* engine);
SL_API int32_t sl_engine_max_block_frames(const sl_engine* engine);

/* Control thread. Queued and applied at the next block. */
SL_API int32_t sl_engine_set_output_gain(sl_engine* engine, float gain);
SL_API int32_t sl_engine_set_master_muted(sl_engine* engine, int32_t muted);

/* Audio side, offline use: process `frames` interleaved frames in place.
 * The buffer is borrowed for this call only. Returns SL_ERR_BUSY while the
 * device thread is running. */
SL_API int32_t sl_engine_process(sl_engine* engine, float* interleaved, int32_t frames);

/* Starts a native thread that stands in for an audio device. Every
 * max_block_frames frames of time it renders a sine at tone_hz (amplitude
 * 0.5), processes it and queues the result for sl_engine_read_output.
 * Phase 4 replaces it with a JUCE AudioDeviceManager callback. */
SL_API int32_t sl_engine_start_device(sl_engine* engine, float tone_hz);
SL_API int32_t sl_engine_stop_device(sl_engine* engine);
SL_API int32_t sl_engine_device_running(const sl_engine* engine);

/* Control thread: copy up to max_frames of queued device output, oldest
 * first, into `out` (max_frames * channels floats). Returns the number of
 * frames copied, or a negative error. About one second is queued; when the
 * reader falls behind, new output is dropped and counted in the stats. */
SL_API int32_t sl_engine_read_output(sl_engine* engine, float* out, int32_t max_frames);

/* Control thread: absolute peak since the last call, then reset to 0. */
SL_API float sl_engine_take_peak(sl_engine* engine);

SL_API int32_t sl_engine_get_stats(const sl_engine* engine, sl_engine_stats* out);

#ifdef __cplusplus
}
#endif

#endif /* SONICLAB_ENGINE_CAPI_H */
