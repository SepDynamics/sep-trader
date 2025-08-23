export interface OrderParams {
  symbol: string;
  side: string;
  quantity: number;
  type: string;
  price?: number;
}

export const buildOrder = ({ symbol, side, quantity, type, price }: OrderParams) => {
  const order: any = {
    symbol,
    side,
    quantity,
    type,
  };
  if (type === 'limit' && typeof price !== 'undefined') {
    order.price = price;
  }
  return order;
};
