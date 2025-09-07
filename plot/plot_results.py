import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
from datetime import datetime

def plot_results(result, save_dir="results"):
    """
    result: solver.solve(...) が返す辞書
    save_dir: 保存先ディレクトリ
    """
    X = np.array(result["X"])
    Xh = np.array(result["Xh"])
    U = np.array(result["U"])

    T = X.shape[0] - 1
    t = np.arange(T+1)

    fig, axs = plt.subplots(3, 1, figsize=(8, 8), sharex=True)

    # --- 状態 ---
    axs[0].plot(t, X[:, 0], label="theta (true)")
    axs[0].plot(t, Xh[:, 0], "--", label="theta (est)")
    axs[0].plot(t, X[:, 1], label="omega (true)")
    axs[0].plot(t, Xh[:, 1], "--", label="omega (est)")
    axs[0].set_ylabel("state")
    axs[0].legend()

    # --- 制御 ---
    axs[1].plot(np.arange(T), U[:, 0], label="u")
    axs[1].set_ylabel("control")
    axs[1].legend()

    # --- コスト履歴（もし保存しているなら）---
    if "cost_trace" in result:
        axs[2].plot(result["cost_trace"], "-o", label="J per iter")
        axs[2].set_ylabel("cost")
        axs[2].set_xlabel("iteration")
        axs[2].legend()
    else:
        axs[2].axis("off")

    fig.suptitle(f"iLQG Results (final cost={result['J']:.3f})")

    # --- 保存処理 ---
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{timestamp}_ilqg_result.png"
    save_path = os.path.join(save_dir, filename)

    plt.savefig(save_path)
    print(f"Saved figure to {save_path}")

    plt.close(fig)

if __name__ == "__main__":
    # solve() の結果を pickle から読み込んで描画する例
    with open("ilqg_result.pkl", "rb") as f:
        result = pickle.load(f)
    plot_results(result)
