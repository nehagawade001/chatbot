INSERT INTO store.products (product_name, product_description, product_category, product_type, product_price, created_at)
VALUES
('Logitech M510', 'Wireless mouse with ergonomic design.', 'Input Devices', 'Mouse', 29.99, NOW()),
('Corsair K95 RGB', 'Mechanical gaming keyboard with customizable RGB lighting.', 'Input Devices', 'Keyboard', 49.99, NOW()),
('Anker PowerCore 10000', 'Portable charger with 10000mAh capacity.', 'Power Accessories', 'Battery', 19.99, NOW()),
('Anker 4-Port USB 3.0 Hub', 'USB hub for connecting multiple devices.', 'Connectivity Accessories', 'USB Hub', 39.99, NOW()),
('Logitech C920', 'HD webcam for video conferencing.', 'Output Devices', 'Webcam', 79.99, NOW());

INSERT INTO store.product_rating_review (product_id, product_rating, product_review)
VALUES
(16, 4.5, 'Great ergonomic mouse, perfect for long hours of use.'),
(17, 4.8, 'Excellent keyboard with responsive keys and customizable lighting.'),
(18, 4.7, 'Reliable power bank, charges my devices quickly.'),
(19, 4.2, 'Convenient USB hub, perfect for connecting multiple devices.'),
(20, 4.6, 'High-quality webcam, very clear image for video calls.');
