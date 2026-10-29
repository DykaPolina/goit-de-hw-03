from pyspark.sql import SparkSession
from pyspark.sql import functions as F

spark = SparkSession.builder.appName("goit-de-hw-03").getOrCreate()

# 1. Завантаження CSV-файлів як окремих DataFrame
users_df = spark.read.csv("users.csv", header=True, inferSchema=True)
purchases_df = spark.read.csv("purchases.csv", header=True, inferSchema=True)
products_df = spark.read.csv("products.csv", header=True, inferSchema=True)

print("p1: users")
users_df.show(5)
print("p1: purchases")
purchases_df.show(5)
print("p1: products")
products_df.show(5)

# 2. Очищення даних: видалення рядків з пропущеними значеннями.
# Згідно з умовою, рядок видаляється, якщо пропущене будь-яке поле
# (зокрема й product_name чи email, які в розрахунках не беруть участі).
rows_before = (users_df.count(), purchases_df.count(), products_df.count())

users_df = users_df.dropna()
purchases_df = purchases_df.dropna()
products_df = products_df.dropna()

print(f"p2: rows before dropna — users: {rows_before[0]}, "
      f"purchases: {rows_before[1]}, products: {rows_before[2]}")
print(f"p2: rows after dropna  — users: {users_df.count()}, "
      f"purchases: {purchases_df.count()}, products: {products_df.count()}")
users_df.show(5)
purchases_df.show(5)
products_df.show(5)

# Єдиний очищений набір для всіх наступних кроків: покупка враховується,
# лише якщо і користувач, і продукт пройшли очищення (inner join).
# Сума покупки = quantity * price.
purchases_full_df = purchases_df \
    .join(users_df, on="user_id", how="inner") \
    .join(products_df, on="product_id", how="inner") \
    .withColumn("total", F.col("quantity") * F.col("price"))

print(f"Purchases with valid user and product: {purchases_full_df.count()} "
      f"of {purchases_df.count()}")

# 3. Загальна сума покупок за кожною категорією продуктів
total_by_category_df = purchases_full_df.groupBy("category") \
    .agg(F.round(F.sum("total"), 2).alias("total_purchases")) \
    .orderBy(F.desc("total_purchases"))

print("p3: total purchases by category")
total_by_category_df.show()

# 4. Сума покупок за категоріями для віку від 18 до 25 включно
purchases_18_25_df = purchases_full_df.filter(F.col("age").between(18, 25))

total_by_category_18_25_df = purchases_18_25_df.groupBy("category") \
    .agg(F.round(F.sum("total"), 2).alias("total_purchases")) \
    .orderBy(F.desc("total_purchases"))

print("p4: total purchases by category, age 18-25")
total_by_category_18_25_df.show()

# 5. Частка покупок кожної категорії від сумарних витрат вікової групи 18-25.
# Округлення виконується лише для фінального результату.
total_18_25 = purchases_18_25_df.agg(F.sum("total")).collect()[0][0]

share_by_category_18_25_df = purchases_18_25_df.groupBy("category") \
    .agg(F.sum("total").alias("category_total")) \
    .withColumn("percentage", F.round(F.col("category_total") / total_18_25 * 100, 2)) \
    .withColumn("category_total", F.round("category_total", 2)) \
    .orderBy(F.desc("percentage"))

print(f"p5: share of purchases by category, age 18-25 (%), "
      f"total spending 18-25 = {round(total_18_25, 2)}")
share_by_category_18_25_df.show()

# 6. Топ-3 категорії з найвищим відсотком витрат для віку 18-25
print("p6: top-3 categories by percentage, age 18-25")
share_by_category_18_25_df.limit(3).show()

spark.stop()
