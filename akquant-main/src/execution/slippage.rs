use crate::model::OrderSide;
use rust_decimal::Decimal;

/// 滑点模型特征
pub trait SlippageModel: Send + Sync {
    /// 计算滑点后的成交价
    fn calculate_price(&self, price: Decimal, quantity: Decimal, side: OrderSide) -> Decimal;
}

/// 零滑点模型 (默认)
#[derive(Debug, Clone, Copy, Default)]
pub struct ZeroSlippage;

impl SlippageModel for ZeroSlippage {
    fn calculate_price(&self, price: Decimal, _quantity: Decimal, _side: OrderSide) -> Decimal {
        price
    }
}

/// 固定值滑点模型
#[derive(Debug, Clone, Copy)]
pub struct FixedSlippage {
    pub delta: Decimal,
}

impl SlippageModel for FixedSlippage {
    fn calculate_price(&self, price: Decimal, _quantity: Decimal, side: OrderSide) -> Decimal {
        match side {
            OrderSide::Buy => price + self.delta,
            OrderSide::Sell => price - self.delta,
        }
    }
}

/// 百分比滑点模型
#[derive(Debug, Clone, Copy)]
pub struct PercentSlippage {
    pub rate: Decimal,
}

impl SlippageModel for PercentSlippage {
    fn calculate_price(&self, price: Decimal, _quantity: Decimal, side: OrderSide) -> Decimal {
        match side {
            OrderSide::Buy => price * (Decimal::ONE + self.rate),
            OrderSide::Sell => price * (Decimal::ONE - self.rate),
        }
    }
}
