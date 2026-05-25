-- =========================================
-- TARGET BRAZIL SQL ANALYSIS
-- PAYMENT ANALYSIS
-- =========================================


-- 1. Month-on-Month Orders by Payment Type

SELECT
    payments.payment_type,

    YEAR(orders.order_purchase_timestamp) AS order_year,

    MONTH(orders.order_purchase_timestamp) AS order_month,

    COUNT(orders.order_id) AS total_orders

FROM payments

JOIN orders
    ON payments.order_id = orders.order_id

GROUP BY
    payments.payment_type,
    order_year,
    order_month

ORDER BY
    payments.payment_type,
    order_year,
    order_month;



-- 2. Orders Based on Payment Installments

SELECT
    payment_installments,

    COUNT(order_id) AS total_orders

FROM payments

GROUP BY payment_installments

ORDER BY payment_installments;



-- 3. Most Frequently Used Payment Methods

SELECT
    payment_type,

    COUNT(order_id) AS total_orders

FROM payments

GROUP BY payment_type

ORDER BY total_orders DESC;



-- 4. Average Payment Value by Payment Type

SELECT
    payment_type,

    ROUND(AVG(payment_value), 2) AS avg_payment_value

FROM payments

GROUP BY payment_type

ORDER BY avg_payment_value DESC;



-- 5. Payment Installment Trends

SELECT
    payment_installments,

    ROUND(AVG(payment_value), 2) AS avg_payment_amount,

    COUNT(order_id) AS total_orders

FROM payments

GROUP BY payment_installments

ORDER BY payment_installments;
