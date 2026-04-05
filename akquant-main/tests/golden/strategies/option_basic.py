"""Option basic strategy."""

from akquant import Bar, Strategy


class OptionBasicStrategy(Strategy):
    """Test Option Trading PnL."""

    def __init__(self) -> None:
        """Initialize."""
        self.count = 0

    def on_bar(self, bar: Bar) -> None:
        """Handle bar data.

        Args:
            bar: Bar data.
        """
        self.count += 1

        # Day 1: Buy 1 contract
        if self.count == 1:
            print(f"Buying 1 contract {bar.symbol} @ {bar.close}")
            self.buy(bar.symbol, 1)

        # Day 5: Sell 1 contract (Close position)
        if self.count == 5:
            print(f"Selling 1 contract {bar.symbol} @ {bar.close}")
            self.sell(bar.symbol, 1)
