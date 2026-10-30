"""Live plotting helpers for Ultralytics coursework training runs."""

from __future__ import annotations


def add_live_metrics_plot(model, every: int = 1) -> None:
    """Refresh results.png from results.csv every N completed epochs.

    Set ``every`` to 0 to leave plotting to Ultralytics' final training hook.
    """
    if every < 0:
        raise ValueError("plot interval must be zero or a positive integer")
    if every == 0:
        return

    from ultralytics.utils.plotting import plot_results

    def refresh_plot(trainer) -> None:
        completed_epochs = trainer.epoch + 1
        if completed_epochs % every and completed_epochs != trainer.epochs:
            return
        try:
            plot_results(file=trainer.csv)
        except Exception as exc:  # Keep monitoring failures from interrupting training.
            print(f"Live metrics plot update failed: {exc}", flush=True)

    model.add_callback("on_fit_epoch_end", refresh_plot)
