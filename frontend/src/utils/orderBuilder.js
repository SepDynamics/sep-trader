export const buildOrder = ({ symbol, side, quantity, type, price }) => ({
  symbol,
  side,
  quantity: parseInt(quantity, 10),
  type,
  ...(type === 'limit' && { price: parseFloat(price) })
});

export default buildOrder;
