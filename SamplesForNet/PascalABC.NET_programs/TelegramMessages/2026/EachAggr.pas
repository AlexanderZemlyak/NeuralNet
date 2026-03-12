type 
  Product = auto class
    Category: string;
    Model: string;    
    Price: real;
    Rating: real;
  end;

begin
  var products := 
    [
      new Product('Phones', 'iPhone 13', 499.99, 4.5),
      new Product('Phones', 'Galaxy S21', 699.99, 4.7),
      new Product('Phones', 'iPhone 13', 299.99, 4.2),  
      new Product('Laptops', 'MacBook Pro', 999.99, 4.8),
      new Product('Laptops', 'MacBook Pro', 1299.99, 4.9), 
      new Product('Tablets', 'iPad Air', 399.99, 4.3),
      new Product('Tablets', 'iPad Air', 399.99, 4.3)   
    ];

  // 1. Количество уникальных моделей по категориям
  var uniqueModels := products
    .GroupBy(p -> p.Category)
    .Each(g -> g.DistinctBy(p -> p.Model).Count);
  
  // 2. Средняя цена по категориям
  var avgPrices := products
    .GroupBy(p -> p.Category)
    .Each(g -> g.Average(p -> p.Price));
  
  // 3. Количество товаров по категориям
  var totalCounts := products
    .GroupBy(p -> p.Category)
    .Each(g -> g.Count);
  
  Println('Уникальные модели по категориям:');
  uniqueModels.PrintLines(kv -> $'  {kv.Key}: {kv.Value}');

  Println('Средняя цена по категориям:');
  avgPrices.PrintLines(kv -> $'  {kv.Key}: ${kv.Value:F2}');

  Println('Всего товаров по категориям:');
  totalCounts.PrintLines(kv -> $'  {kv.Key}: {kv.Value}');
end.