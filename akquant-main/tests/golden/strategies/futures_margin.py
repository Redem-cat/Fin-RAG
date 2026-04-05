"""Futures margin strategy."""

from typing import Any

from akquant import Bar, OrderStatus, Strategy


class FutureMarginStrategy(Strategy):
    """Test Futures Margin Logic."""

    def __init__(self) -> None:
        """Initialize."""
        self.count = 0

    def on_bar(self, bar: Bar) -> None:
        """Handle bar data.

        Args:
            bar: Bar data.
        """
        self.count += 1

        # Day 1: Buy 1 lot
        if self.count == 1:
            print(f"Buying 1 lot {bar.symbol} @ {bar.close}")
            self.buy(bar.symbol, 1)

        # Day 2: Buy another lot (Should fail due to insufficient margin)
        elif self.count == 2:
            print(f"Attempting to Buy 1 more lot {bar.symbol} @ {bar.close}")
            self.buy(bar.symbol, 1)

    def on_order(self, order: Any) -> None:
        """Handle order updates.

        Args:
            order: Order object.
        """
        if order.status == OrderStatus.Rejected:
            print(f"Order Rejected: {order.reject_reason}")
