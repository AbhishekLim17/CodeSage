package shop;

import java.util.List;

public class OrderService {
    private final List<String> orders;

    public OrderService(List<String> orders) {
        this.orders = orders;
    }

    /** Place an order. */
    public void place(String item) {
        orders.add(item);
    }

    interface Listener {
        void onPlaced(String item);
    }
}
