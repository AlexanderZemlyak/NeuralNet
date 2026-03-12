{$reference Mathnet.Numerics.dll}
uses Mathnet.Numerics;
uses Coords;

function GenerateCluster(X, Y, deviation: real; count: integer): array of Point;
begin
  var xx := Generate.Normal(count, X, deviation);
  var yy := Generate.Normal(count, Y, deviation);
  Result := ArrGen(count, i -> Pnt(xx[i],yy[i]));
end;

function Distance(p1, p2: Point): real := Sqrt(Sqr(p1.x - p2.x) + Sqr(p1.y - p2.y));

function GenerateRandomCentroids(k: integer; points: array of Point; minDistance: real): array of Point;
begin
  var minX := points.Min(p -> p.x);
  var maxX := points.Max(p -> p.x);
  var minY := points.Min(p -> p.y);
  var maxY := points.Max(p -> p.y);

  var centroids := new List<Point>();
  
  while centroids.Count < k do
  begin
    // Случайная точка в пределах прямоугольника
    var candidate := new Point(
      RandomReal(minX, maxX),
      RandomReal(minY, maxY)
    );

    // Проверяем, что расстояние до всех ранее выбранных центроидов больше minDistance
    var isFarEnough := centroids.All(c -> Distance(c,candidate) >= minDistance);

    if isFarEnough then
      centroids.Add(candidate);
  end;

  Result := centroids.ToArray();
end;

function KMeans(points: array of Point; k: integer; maxIterations: integer; minDistance: real): (array of List<Point>, array of Point);
begin
  var n := points.Length;
  
  if k > n then
    raise new System.ArgumentException('Число кластеров не может быть больше числа точек');

  // Генерируем начальные центроиды случайно в пределах прямоугольника
  var centroids := GenerateRandomCentroids(k, points, minDistance);
  
  // Инициализация кластеров как массив списков с помощью ArrGen
  var clusters := ArrGen(k, i -> new List<Point>);

  for var iteration := 1 to maxIterations do
  begin
    // Шаг 1: Назначаем каждую точку ближайшему центроиду
    foreach var cluster in clusters do
      cluster.Clear(); // Очищаем списки на каждой итерации
    
    foreach var point in points do
    begin
      var nearestCentroidIndex := centroids.IndexMinBy(c -> Distance(point,c));
      clusters[nearestCentroidIndex].Add(point); // Добавляем точку в ближайший кластер
    end;

    // Шаг 2: Обновляем центроиды как средние точки своих кластеров
    var newCentroids := Copy(centroids); // Копируем центроиды
    
    for var i := 0 to k - 1 do
      if clusters[i].Count > 0 then
        newCentroids[i] := new Point(
          clusters[i].Average(p -> p.x),
          clusters[i].Average(p -> p.y)
        );

    // Проверяем, если центроиды не изменились, то завершаем
    if newCentroids.SequenceEqual(centroids) then
      break;

    centroids := newCentroids;
  end;

  Result := (clusters, centroids);
end;
begin
  // Сгенерируем исходные кластеры точек
  var cluster1 := GenerateCluster(5, 3, 2.5, 50);
  var cluster2 := GenerateCluster(7, -6, 1.5, 25);
  var cluster3 := GenerateCluster(-7, 2, 1, 44);
  
  var allPoints := cluster1 + cluster2 + cluster3;
  
  // Применяем алгоритм K-means для кластеризации
  var (clusters, centroids) := KMeans(allPoints, 3, 100, 2);
  
  DrawPoints(clusters[0].ToArray,3);
  DrawPoints(clusters[1].ToArray,3);
  DrawPoints(clusters[2].ToArray,PointRadius := 3);

  DrawPoint(centroids[0].x,centroids[0].y,PointRadius := 5);
  DrawPoint(centroids[1].x,centroids[1].y,PointRadius := 5);
  DrawPoint(centroids[2].x,centroids[2].y,PointRadius := 5);
end.