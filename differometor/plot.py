import os
import numpy as np
import matplotlib.pyplot as plt
from differometor.components import HARD_SIDE_POWER_THRESHOLD, SOFT_SIDE_POWER_THRESHOLD, DETECTOR_POWER_THRESHOLD


def plot_comparison(
        x, 
        y_1, 
        y_2=None, 
        plot_directory=".", 
        name="sensitivity", 
        y_1_name='optimization', 
        y_2_name='baseline'
    ):
    os.makedirs(plot_directory, exist_ok=True)

    plt.figure()
    if y_2 is not None:
        plt.plot(x, y_2, label=y_2_name, linestyle='--', marker='o', markersize=2, linewidth=3)
    plt.plot(x, y_1, label=y_1_name, linestyle='-', marker='o', markersize=2, linewidth=1)

    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Strain Sensitivity [1/sqrt(Hz)]")
    plt.yscale('log')
    plt.xscale('log')
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{plot_directory}/{name}.png")
    plt.close()


def plot_best_losses(
        best_losses, 
        folder, 
        power_violations, 
        suffix=""
    ):
    best_losses = np.asarray(best_losses, dtype=float)
    power_violations = np.asarray(power_violations, dtype=bool)
    n = best_losses.shape[0]
    idx = np.arange(n)
    viol_mask = power_violations
    ok_mask = ~power_violations

    arg_min = int(np.nanargmin(best_losses))
    min_val = float(best_losses[arg_min])
    mean_val = float(np.nanmean(best_losses))

    plt.figure()
    plt.scatter(idx[viol_mask], best_losses[viol_mask], marker='x', color="blue", label="Violating")
    plt.scatter(idx[ok_mask], best_losses[ok_mask], marker='o', color="blue", label="Non violating")
    min_marker = 'x' if power_violations[arg_min] else 'o'
    plt.scatter([arg_min], [min_val], color='r', marker=min_marker, label=f"Best: {arg_min}")
    plt.axhline(y=mean_val, linestyle='--', color='black', label='Mean Losses')

    plt.ylabel('best loss')
    plt.xlabel('run')
    plt.legend()
    plt.tight_layout()
    plt.ylim(bottom=0 if min_val > 0 else min_val * 1.1)
    plt.grid(True, linestyle=':', linewidth=0.7)
    plt.savefig(f"{folder}/best_losses{suffix}.png")
    plt.close()


def plot_loss_curve(
        losses, 
        folder
    ):
    os.makedirs(folder, exist_ok=True)

    losses = np.asarray(losses, dtype=float)
    x_full = np.arange(len(losses))

    # Keep only finite values for plotting
    finite_mask = np.isfinite(losses)
    if not finite_mask.any():
        return
    losses = losses[finite_mask]
    x_full = x_full[finite_mask]

    def filter_large_jumps(arr, xs):
        arr = np.asarray(arr)
        xs = np.asarray(xs)
        if arr.size == 0:
            return arr, xs
        filtered = [arr[0]]
        filtered_x = [xs[0]]
        for i in range(1, len(arr)):
            if arr[i] <= 2 * filtered[-1]:
                filtered.append(arr[i])
                filtered_x.append(xs[i])
        return np.asarray(filtered), np.asarray(filtered_x)

    for loss_scale in ['lin', 'log']:
        for loss_type in ["", "smoothed"]:
            if loss_type == "smoothed":
                y, x = filter_large_jumps(losses, x_full)
            else:
                y, x = losses, x_full

            if y.size == 0:
                continue

            plt.figure()
            plt.plot(x, y, label=f"{loss_type or 'raw'}")
            plt.ylabel('loss')
            plt.xlabel('iteration')
            plt.legend()
            plt.grid(True)

            if loss_scale == 'log':
                # If any non-positive values exist, use symmetrical log scale.
                if np.any(y <= 0):
                    # Choose a linear region around zero so negatives/zero are visible.
                    # Use the smallest non-zero |y| as a guide.
                    nonzero = np.abs(y[np.nonzero(y)])
                    linthresh = float(np.nanmin(nonzero)) if nonzero.size else 1e-8
                    linthresh = max(linthresh, 1e-8)
                    plt.yscale('symlog', linthresh=linthresh, linscale=1.0, base=10)
                    suffix = "log_symlog"
                else:
                    plt.yscale('log')
                    suffix = "log"
                plt.tight_layout()
                plt.savefig(f"{folder}/{suffix}_losses_{loss_type or 'raw'}.png", dpi=200)
            else:
                plt.tight_layout()
                plt.savefig(f"{folder}/losses_{loss_type or 'raw'}.png", dpi=200)

            plt.close()


def plot_powers(
        hard_side_powers, 
        soft_side_powers, 
        detector_powers, 
        suffix, 
        folder
    ):
    def plot_bar_diagram(powers, cutoff, name):
        plt.figure()
        plt.bar(np.arange(len(powers)), powers.squeeze())
        plt.axhline(y=cutoff, color='r', linestyle='--')
        plt.ylabel('Power [W]')
        plt.xlabel('component')
        plt.yscale('log')
        plt.tight_layout()
        plt.grid()
        plt.savefig(f"{folder}/powers_{name}{suffix}.png")
        plt.close()

    plot_bar_diagram(hard_side_powers, HARD_SIDE_POWER_THRESHOLD, "hard_side")
    plot_bar_diagram(soft_side_powers, SOFT_SIDE_POWER_THRESHOLD, "soft_side")
    plot_bar_diagram(detector_powers, DETECTOR_POWER_THRESHOLD, "detector")
