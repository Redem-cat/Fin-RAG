"""Stock T+1 strategy."""

from typing import Any

from akquant import Bar, OrderStatus, Strategy


class StockT1Strategy(Strategy):
    """Test T+1 rule and Limit Up/Down execution."""

    def __init__(self) -> None:
        """Initialize."""
        self.day_count = 0
        self.order_ids = {}  # type: ignore

    def on_bar(self, bar: Bar) -> None:
        """Handle bar data.

        Args:
            bar: Bar data.
        """
        self.day_count += 1

        # Day 1: Buy 100
        if self.day_count == 1:
            print(f"Day 1: Buying 100 {bar.symbol}")
            self.buy(bar.symbol, 100)

        # Day 2: Sell 100 (Should succeed as it is T+1 from Day 1)
        elif self.day_count == 2:
            print(f"Day 2: Selling 100 {bar.symbol}")
            self.sell(bar.symbol, 100)

        # Day 3: Buy 100, then immediately try to Sell 100 (Intraday T+0 attempt)
        elif self.day_count == 3:
            print(f"Day 3: Buying 100 {bar.symbol}")
            self.buy(bar.symbol, 100)

            print(f"Day 3: Attempting to Sell 100 {bar.symbol} (Should fail T+1)")
            self.sell(bar.symbol, 100)

        # Day 4: Sell 100 (Should succeed as it is T+1 from Day 3)
        elif self.day_count == 4:
            print(f"Day 4: Selling 100 {bar.symbol}")
            self.sell(bar.symbol, 100)

    def on_order(self, order: Any) -> None:
        """Handle order updates.

        Args:
            order: Order object.
        """
        if order.status == OrderStatus.Rejected:
            print(f"Order Rejected: {order.reject_reason}")
