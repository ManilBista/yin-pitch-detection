
# import numpy as np
# import scipy.signal as sg


# # ----------------------------------------------------------------------------
# # CELL 1 (verbatim)
# # ----------------------------------------------------------------------------
# def difference_vectorized(x, W, max_tau):
#     diff = np.zeros(max_tau + 1)
#     for tau in range(max_tau + 1):
#         # Shift and subtract
#         diff[tau] = np.sum((x[1:W-tau] - x[1+tau:W])**2)
#     return diff


# # ----------------------------------------------------------------------------
# # CELL 2 (verbatim)
# # ----------------------------------------------------------------------------
# def computeCmndf(x, W, min_tau, max_tau):
#     diff = difference_vectorized(x, W, max_tau)
#     cmndf = np.empty(max_tau - min_tau)

#     # Must accumulate from j=1, not from min_tau
#     running_sum = 0.0
#     for j in range(1, min_tau):
#         running_sum += diff[j]          # build up sum before the window

#     for tau in range(min_tau, max_tau):
#         running_sum += diff[tau]        # now add current tau (once, correctly)
#         cumulative_mean = running_sum / tau
#         # cmndf[tau - min_tau] = diff[tau] / cumulative_mean
#         if cumulative_mean == 0:
#             cmndf[tau - min_tau] = 1.0  # treat as unvoiced
#         else:
#             cmndf[tau - min_tau] = diff[tau] / cumulative_mean

#     return cmndf


# # ----------------------------------------------------------------------------
# # CELL 3 (verbatim)
# # ----------------------------------------------------------------------------
# def find_first_local_min_below_threshold(array, threshold):
#     # Find all local minima
#     local_minima_indices = sg.argrelmin(array)[0]
#     for idx in local_minima_indices:
#         # Return the FIRST dip that drops below threshold
#         if array[idx] < threshold:
#             return idx
#     return None


# # ----------------------------------------------------------------------------
# # CELL 4 (verbatim)
# # ----------------------------------------------------------------------------
# def parabolic_interp(y1, y2, y3):
#     # Finds the fractional offset for the exact bottom of a parabola
#     return 0.5 * (y1 - y3) / (y1 - 2*y2 + y3)


# # ----------------------------------------------------------------------------
# # CELL 5 (verbatim)
# # ----------------------------------------------------------------------------
# def octave_correct(cmndf, predicted_idx, min_tau, threshold):
#     """Check whether the true fundamental is one octave below the candidate.

#     If the candidate is a harmonic (e.g. 2nd harmonic at 220 Hz when the
#     fundamental is 110 Hz), the CMNDF will also have a valid dip at
#     approximately double the lag.  If that dip is below a relaxed threshold,
#     we prefer the longer lag (lower frequency = fundamental).
#     """
#     # Index in the cmndf array that corresponds to 2× the candidate lag
#     doubled_idx = min_tau + 2 * predicted_idx
#     relaxed_thresh = min(threshold * 2.0, 0.8)  # allow some slack

#     if doubled_idx >= len(cmndf):
#         return predicted_idx

#     # Check a small neighbourhood (±2) around the doubled index
#     best_idx = doubled_idx
#     best_val = cmndf[doubled_idx]
#     for offset in range(-2, 3):
#         ni = doubled_idx + offset
#         if 0 <= ni < len(cmndf) and cmndf[ni] < best_val:
#             best_val = cmndf[ni]
#             best_idx = ni

#     if best_val < relaxed_thresh:
#         return best_idx
#     return predicted_idx


# # ----------------------------------------------------------------------------
# # CELL 5 (verbatim)
# # ----------------------------------------------------------------------------
# def pitchDetect(audio, fs, min_f0, max_f0, W, decimation_factor, cmndf_threshold,
#                 rms_threshold=0.05):
#     res = []

#     # Downsample for speed
#     downsampled_audio = sg.decimate(audio, decimation_factor, zero_phase=True)
#     downsampled_fs = fs // decimation_factor

#     # Calculate bounds based on DOWNSAMPLED frequency
#     min_tau = downsampled_fs // max_f0
#     max_tau = downsampled_fs // min_f0

#     # 50% Overlapping frames
#     step = W // 2
#     length = (len(downsampled_audio) // step - 1) * step

#     for start in range(0, length, step):
#         x = downsampled_audio[start:start+W]
#         if len(x) != W:
#             break

#         # RMS energy gate — check BEFORE appending anything
#         frame_energy = np.sqrt(np.mean(x**2))
#         if frame_energy < rms_threshold:
#             res.append(None)
#             continue

#         cmndf = computeCmndf(x, W, min_tau, max_tau)
#         predicted_idx = find_first_local_min_below_threshold(cmndf, cmndf_threshold)

#         if predicted_idx is not None:
#             # Octave correction — prefer fundamental over harmonics
#             predicted_idx = octave_correct(cmndf, predicted_idx, min_tau, cmndf_threshold)

#             # Safety check so interpolation doesn't crash at the edges
#             if 0 < predicted_idx < len(cmndf) - 1:
#                 y1 = cmndf[predicted_idx - 1]
#                 y2 = cmndf[predicted_idx]
#                 y3 = cmndf[predicted_idx + 1]

#                 interp_add = parabolic_interp(y1, y2, y3)

#                 # Convert array index back to actual lag
#                 actual_tau = min_tau + predicted_idx + interp_add
#                 f0 = downsampled_fs / actual_tau
#                 res.append(f0)
#             else:
#                 res.append(None)
#         else:
#             res.append(None)  # Unvoiced

#     return np.array(res)








import numpy as np
import scipy.signal as sg

def _next_power_of_2(n):
    p = 1
    while p < n:
        p <<= 1
    return p


def difference_vectorized(x, W, max_tau):
    # FFT-based difference function: O(W log W) instead of O(W * max_tau)
    #
    # diff(tau) = sum_{j=1}^{W-tau-1} (x[j] - x[j+tau])^2
    #           = sum(x[j]^2, j=1..W-tau-1) + sum(x[j+tau]^2, j=1..W-tau-1)
    #             - 2 * sum(x[j]*x[j+tau], j=1..W-tau-1)
    
    # The cross-correlation term is computed via FFT autocorrelation.
    # The energy terms use a precomputed cumulative sum of x^2.

    n_fft = _next_power_of_2(2 * W)
    x_pad = np.zeros(n_fft)
    x_pad[:W] = x

    # Autocorrelation via FFT: acf[tau] = sum(x[j]*x[j+tau], j=0..W-1-tau)
    X = np.fft.rfft(x_pad)
    acf = np.fft.irfft(X * np.conj(X), n_fft)[:W]

    # Cumulative sum of x^2 for fast energy lookups
    x_sq_cs = np.zeros(W + 1)
    np.cumsum(x ** 2, out=x_sq_cs[1:])   # x_sq_cs[i] = sum(x[0:i]^2)

    taus = np.arange(max_tau + 1)

    # Energy: sum(x[j]^2, j=1..W-tau-1) = x_sq_cs[W-tau] - x_sq_cs[1]
    e1 = x_sq_cs[W - taus] - x_sq_cs[1]

    # Energy: sum(x[j+tau]^2, j=1..W-tau-1) = sum(x[k]^2, k=1+tau..W-1)
    #        = x_sq_cs[W] - x_sq_cs[1+tau]
    e2 = x_sq_cs[W] - x_sq_cs[1 + taus]

    # Cross-correlation (starting from j=1): acf[tau] - x[0]*x[tau]
    cross = acf[:max_tau + 1] - x[0] * x[:max_tau + 1]

    diff = e1 + e2 - 2.0 * cross
    return diff


def computeCmndf(x, W, min_tau, max_tau):
    diff = difference_vectorized(x, W, max_tau)

    # Vectorized cumulative sum replaces both Python for-loops.
    # diff[0] = 0, so cumsum from 0 == cumsum from 1.
    # running_sum at tau = sum(diff[1..tau]) = cumsum_all[tau]
    cumsum_all = np.cumsum(diff)

    taus = np.arange(min_tau, max_tau)
    running_sums = cumsum_all[taus]
    cumulative_means = running_sums / taus

    # Avoid division by zero (same as the original if-check)
    cmndf = np.where(cumulative_means == 0, 1.0,
                      diff[taus] / cumulative_means)

    return cmndf


def find_first_local_min_below_threshold(array, threshold):
    # Find all local minima
    local_minima_indices = sg.argrelmin(array)[0]
    for idx in local_minima_indices:
        # Return the FIRST dip that drops below threshold
        if array[idx] < threshold:
            return idx
    return None


def parabolic_interp(y1, y2, y3):
    # Finds the fractional offset for the exact bottom of a parabola
    return 0.5 * (y1 - y3) / (y1 - 2*y2 + y3)


def octave_correct(cmndf, predicted_idx, min_tau, threshold):

    candidate_val = cmndf[predicted_idx]

    # Index in the cmndf array that corresponds to 2× the candidate lag
    doubled_idx = min_tau + 2 * predicted_idx
    relaxed_thresh = min(threshold * 2.0, 0.8)  # allow some slack

    if doubled_idx >= len(cmndf):
        return predicted_idx

    # Check a small neighbourhood (±2) around the doubled index
    best_idx = doubled_idx
    best_val = cmndf[doubled_idx]
    for offset in range(-2, 3):
        ni = doubled_idx + offset
        if 0 <= ni < len(cmndf) and cmndf[ni] < best_val:
            best_val = cmndf[ni]
            best_idx = ni

    
    # Only correct if sub-harmonic is BOTH stronger than candidate AND
    # below the original (not relaxed) threshold
    if best_val < candidate_val and best_val < threshold:
        return best_idx
    return predicted_idx

    


def pitchDetect(audio, fs, min_f0, max_f0, W, decimation_factor, cmndf_threshold,
                rms_threshold=0.05):
    res = []

    # Downsample for speed
    downsampled_audio = sg.decimate(audio, decimation_factor, zero_phase=True)
    downsampled_fs = fs // decimation_factor

    # Calculate bounds based on DOWNSAMPLED frequency
    min_tau = downsampled_fs // max_f0
    max_tau = downsampled_fs // min_f0

    # 50% Overlapping frames
    step = W // 2
    length = (len(downsampled_audio) // step - 1) * step

    for start in range(0, length, step):
        x = downsampled_audio[start:start+W]
        if len(x) != W:
            break

        # RMS energy gate — check BEFORE appending anything
        frame_energy = np.sqrt(np.mean(x**2))
        if frame_energy < rms_threshold:
            res.append(None)
            continue

        cmndf = computeCmndf(x, W, min_tau, max_tau)
        predicted_idx = find_first_local_min_below_threshold(cmndf, cmndf_threshold)

        if predicted_idx is not None:
            # Octave correction — prefer fundamental over harmonics
            predicted_idx = octave_correct(cmndf, predicted_idx, min_tau, cmndf_threshold)

            # Safety check so interpolation doesn't crash at the edges
            if 0 < predicted_idx < len(cmndf) - 1:
                y1 = cmndf[predicted_idx - 1]
                y2 = cmndf[predicted_idx]
                y3 = cmndf[predicted_idx + 1]

                interp_add = parabolic_interp(y1, y2, y3)

                # Convert array index back to actual lag
                actual_tau = min_tau + predicted_idx + interp_add
                f0 = downsampled_fs / actual_tau
                res.append(f0)
            else:
                res.append(None)
        else:
            res.append(None)  # Unvoiced

    return np.array(res)