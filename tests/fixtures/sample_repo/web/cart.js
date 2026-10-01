import { fetchPrices } from './api.js';

const TAX_RATE = 0.08;

/**
 * Sum the price of every item in the cart.
 */
export function cartTotal(items) {
  return items.reduce((sum, item) => sum + item.price * item.qty, 0);
}

export const applyDiscount = (total, code) => {
  if (code === 'SAVE10') {
    return total * 0.9;
  }
  return total;
};

const formatPrice = async (value) => {
  const prices = await fetchPrices();
  return `${prices.symbol}${value.toFixed(2)}`;
};

export default class Cart {
  constructor() {
    this.items = [];
  }

  add(item) {
    this.items.push(item);
  }

  total() {
    return cartTotal(this.items) * (1 + TAX_RATE);
  }
}
