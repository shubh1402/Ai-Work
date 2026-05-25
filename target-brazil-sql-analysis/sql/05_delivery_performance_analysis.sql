-- =========================================
-- TARGET BRAZIL SQL ANALYSIS
-- DELIVERY PERFORMANCE ANALYSIS
-- =========================================


-- 1. Delivery Time and Estimated Delivery Difference

SELECT
    order_id,

    DATEDIFF(
        order_delivered_customer_date,
        order_purchase_timestamp
    ) AS time_to_deliver,

    DATEDIFF(
        order_delivered_customer_date,
        order_estimated_delivery_date
    ) AS diff_estimated_delivery

FROM orders

WHERE order_delivered_customer_date IS NOT NULL
    AND order_estimated_delivery_date IS NOT NULL;



-- 2. Top 5 States with Highest Average Freight Value

SELECT
    customers.customer_state,

    AVG(order_items.freight_value) AS avg_freight_value

FROM order_items

JOIN orders
    ON order_items.order_id = orders.order_id

JOIN customers
    ON orders.customer_id = customers.customer_id

GROUP BY customers.customer_state

ORDER BY avg_freight_value DESC

LIMIT 5;



-- 3. Top 5 States with Lowest Average Freight Value

SELECT
    customers.customer_state,

    AVG(order_items.freight_value) AS avg_freight_value

FROM order_items

JOIN orders
    ON order_items.order_id = orders.order_id

JOIN customers
    ON orders.customer_id = customers.customer_id

GROUP BY customers.customer_state

ORDER BY avg_freight_value ASC

LIMIT 5;



-- 4. Top 5 States with Highest Average Delivery Time

SELECT
    customers.customer_state,

    AVG(
        DATEDIFF(
            orders.order_delivered_customer_date,
            orders.order_purchase_timestamp
        )
    ) AS avg_delivery_time

FROM orders

JOIN customers
    ON orders.customer_id = customers.customer_id

WHERE orders.order_delivered_customer_date IS NOT NULL

GROUP BY customers.customer_state

ORDER BY avg_delivery_time DESC

LIMIT 5;



-- 5. Top 5 States with Lowest Average Delivery Time

SELECT
    customers.customer_state,

    AVG(
        DATEDIFF(
            orders.order_delivered_customer_date,
            orders.order_purchase_timestamp
        )
    ) AS avg_delivery_time

FROM orders

JOIN customers
    ON orders.customer_id = customers.customer_id

WHERE orders.order_delivered_customer_date IS NOT NULL

GROUP BY customers.customer_state

ORDER BY avg_delivery_time ASC

LIMIT 5;



-- 6. States with Fastest Delivery Compared to Estimated Date

SELECT
    customers.customer_state,

    AVG(
        DATEDIFF(
            orders.order_delivered_customer_date,
            orders.order_estimated_delivery_date
        )
    ) AS avg_delivery_difference

FROM orders

JOIN customers
    ON orders.customer_id = customers.customer_id

WHERE orders.order_delivered_customer_date IS NOT NULL
    AND orders.order_estimated_delivery_date IS NOT NULL

GROUP BY customers.customer_state

ORDER BY avg_delivery_difference ASC

LIMIT 5;
