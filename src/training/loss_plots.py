# pip install matplotlib
import os
import time
import queue
import threading
import tkinter as tk
from dataclasses import dataclass, field
import matplotlib
matplotlib.use("TkAgg")  # set backend BEFORE importing pyplot/figure
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


@dataclass
class TkLossPlotter:
    """Live GUI plotter for training/validation loss.

    Usage:
        plotter = TkLossPlotter()
        plotter.start("My Run")
        plotter.update(0.123)                  # per-batch training loss (running avg)
        plotter.update((0.12, 0.15))           # per-epoch (train, val)
        plotter.snapshot("plots/live.png")     # save any time (non-blocking)
        plotter.stop(save_snapshot=True)       # final save + close
    """
    width: int = 900
    height: int = 500
    refresh_hz: int = 10

    # internal state (do not pass)
    _q_training: queue.Queue = field(default_factory=queue.Queue, init=False)
    _q_validation: queue.Queue = field(default_factory=queue.Queue, init=False)
    _q_cmd: queue.Queue = field(default_factory=queue.Queue, init=False)
    _thread: threading.Thread = field(default=None, init=False)
    _stop: threading.Event = field(default_factory=threading.Event, init=False)
    _training_losses: list = field(default_factory=list, init=False)
    _validation_losses: list = field(default_factory=list, init=False)
    _final_save_path: str | None = field(default=None, init=False)
    _title: str = field(default="Loss Tracker", init=False)

    # ---------- public API ----------
    def start(self, title: str = "Loss Tracker") -> None:
        self._title = title
        if self._thread and self._thread.is_alive():
            return
        self._thread = threading.Thread(target=self._ui_thread, daemon=True)
        self._thread.start()

    def update(self, loss: float | tuple[float, float], training=True) -> None:
        """Non-blocking. Accepts float (train) or (train, val)."""
        if isinstance(loss, tuple):
            tr, va = loss
            self._q_training.put(float(tr))
            if va is not None:
                self._q_validation.put(float(va))
        else:
            if training:
                self._q_training.put(float(loss))
            else:
                self._q_validation.put(float(loss))

    def snapshot(self, save_dir: str | None = None) -> None:
        """Request a save on the UI thread (non-blocking)."""
        # Format title
        bad_chars = ['/', '\\', ':', '*', '?', '"', '<', '>', '|', ' ', '.']
        for char in bad_chars:
            self._title = self._title.replace(char, '_')
        
        # Create directory
        if save_dir:
            save_dir = save_dir + '/' if save_dir[-1] != '/' else save_dir
        else:
            save_dir = "./loss_plots/"
        os.makedirs(save_dir, exist_ok=True)

        # Save path
        self._final_save_path = f"{save_dir}/{time.strftime('%Y%m%d_%H%M%S')}.png"
        self._q_cmd.put(("save", self._final_save_path))

    def stop(self, save_dir: str | None = None) -> None:
        """Finalize: optional final save, then close window.
        save_dir: directory to save the plot to. If None, the plot will be saved to the current working directory.
                    ex: save_dir='./loss_plots/model_tuning/AudioUNet_v1'
        """
        if save_dir:
            self.snapshot(save_dir)

        self._stop.set()
        if self._thread and self._thread.is_alive():
            self._thread.join()  # wait for UI to save & close

    # ---------- UI thread ----------
    def _ui_thread(self) -> None:
        root = tk.Tk()
        root.title(self._title)
        root.geometry(f"{self.width}x{self.height}")

        fig = Figure(figsize=(self.width / 100, self.height / 100), dpi=100)
        ax = fig.add_subplot(111)
        ax.set_title(self._title)
        ax.set_xlabel("Steps")
        ax.set_ylabel("Loss")
        ax.grid(True)

        (line_tr,) = ax.plot([], [], lw=1.8, label="Train")
        (line_va,) = ax.plot([], [], lw=1.8, label="Val")
        ax.legend(loc="upper right")

        canvas = FigureCanvasTkAgg(fig, master=root)
        canvas.get_tk_widget().pack(fill="both", expand=True)

        last_draw = 0.0
        interval = 1.0 / max(1, self.refresh_hz)

        def _sync_and_draw(hard: bool = False) -> None:
            if self._training_losses:
                x_tr = range(1, len(self._training_losses) + 1)
                line_tr.set_data(x_tr, self._training_losses)
            if self._validation_losses:
                x_va = range(1, len(self._validation_losses) + 1)
                line_va.set_data(x_va, self._validation_losses)
            ax.relim()
            ax.autoscale_view()
            (canvas.draw if hard else canvas.draw_idle)()

        def _do_save(path: str) -> None:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            _sync_and_draw(hard=True)  # ensure latest data is on canvas
            fig.savefig(path, bbox_inches="tight", dpi=150)

        def pump_queue() -> None:
            nonlocal last_draw
            changed = False

            # 1) Handle UI commands (e.g., snapshot requests)
            try:
                while True:
                    cmd, arg = self._q_cmd.get_nowait()
                    if cmd == "save":
                        _do_save(arg)
            except queue.Empty:
                pass

            # 2) Drain training queue
            try:
                while True:
                    self._training_losses.append(self._q_training.get_nowait())
                    changed = True
            except queue.Empty:
                pass

            # 3) Drain validation queue
            try:
                while True:
                    self._validation_losses.append(self._q_validation.get_nowait())
                    changed = True
            except queue.Empty:
                pass

            # 4) Refresh at most refresh_hz
            now = time.time()
            if changed and (now - last_draw) >= interval:
                _sync_and_draw(hard=False)
                last_draw = now

            # 5) Finalize or schedule next tick
            if not self._stop.is_set():
                root.after(30, pump_queue)
            else:
                _sync_and_draw(hard=True)
                if self._final_save_path:
                    _do_save(self._final_save_path)
                root.destroy()

        root.after(10, pump_queue)
        root.mainloop()


# --- tiny demo ---
if __name__ == "__main__":
    import math
    plotter = TkLossPlotter()
    plotter.start("LossPlotter Demo")
    try:
        for step in range(1, 301):
            train_loss = 1.0 / step + 0.05 * math.sin(step / 8)
            val_loss = 1.1 / step + 0.05 * math.cos(step / 10)
            plotter.update((train_loss, val_loss))
            time.sleep(0.03)
            if step % 100 == 0:
                plotter.snapshot()  # save mid-run
    finally:
        plotter.stop(save_snapshot=True)
