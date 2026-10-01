export interface Product {
  id: string;
  price: number;
}

export type ProductMap = Record<string, Product>;

export enum Status {
  Active,
  Retired,
}

export class Catalog {
  private products: ProductMap = {};

  add(product: Product): void {
    this.products[product.id] = product;
  }

  find(id: string): Product | undefined {
    return this.products[id];
  }
}
