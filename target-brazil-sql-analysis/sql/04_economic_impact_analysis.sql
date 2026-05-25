-- =========================================
-- TARGET BRAZIL SQL ANALYSIS
-- ECONOMIC IMPACT ANALYSIS
-- =========================================


-- 1. Percentage Increase in Order Cost
-- Comparing Jan-Aug 2017 vs Jan-Aug 2018

SELECT
    (
        (
            SUM(
                CASE
                    WHEN YEAR(orders.order_purchase_timestamp) = 2018
                    THEN payments.payment_value
                    ELSE 0
                END
            )
            -
            SUM(
                CASE
                    WHEN YEAR(orders.order_purchase_timestamp) = 2017
                    THEN payments.payment_value
                    ELSE 0
                END
            )
        )
        /
        SUM(
            CASE
                WHEN YEAR(orders.order_purchase_timestamp) = 2017
                THEN payments.payment_value
                ELSE 0
            END
        )
    ) * 100 AS percentage_increase

FROM payments

JOIN orders
    ON payments.order_id = orders.order_id

WHERE YEAR(orders.order_purchase_timestamp) IN (2017, 2018)
    AND MONTH(orders.order_purchase_timestamp) BETWEEN 1 AND 8;



-- 2. Total and Average Order Value by State

SELECT
    customers.customer_state,

    SUM(payments.payment_value) AS total_order_value,

    AVG(payments.payment_value) AS avg_order_value

FROM payments

JOIN orders
    ON payments.order_id = orders.order_id

JOIN customers
    ON orders.customer_id = customers.customer_id

GROUP BY customers.customer_state

ORDER BY total_order_value DESC;



-- 3. Total and Average Freight Value by State

SELECT
    customers.customer_state,

    SUM(order_items.freight_value) AS total_freight_value,

    AVG(order_items.freight_value) AS avg_freight_value

FROM order_items

JOIN orders
    ON order_items.order_id = orders.order_id

JOIN customers
    ON orders.customer_id = customers.customer_id

GROUP BY customers.customer_state

ORDER BY total_freight_value DESC;
