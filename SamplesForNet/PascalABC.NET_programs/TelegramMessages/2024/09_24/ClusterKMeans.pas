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

function KMeans(points: array of Point; k: integer; maxIterations: integer): (array of array of Point, array of Point);
begin
  // Инициализируем центроиды (случайные точки из исходного набора)
  var centroids := ArrGen(k, i -> points[Random(points.Length)]);
  var clusters := ArrGen(k, i -> new List<Point>()); // Массив для хранения точек в кластерах

  for var iter := 1 to maxIterations do
  begin
    // Очистка предыдущих кластеров
    for var i := 0 to k - 1 do
      clusters[i].Clear;

    // Шаг 1: Назначаем каждой точке ближайший центроид и помещаем её в кластер
    for var i := 0 to points.Length - 1 do
    begin
      var nearestCentroidIndex := centroids.IndexMinBy(c -> point.DistanceTo(c));
      clusters[nearestCentroidIndex].Add(point); // Добавляем точку в ближайший кластер
    end;
    
    // Шаг 2: Пересчитываем центроиды
    for var i := 0 to k - 1 do
      if clusters[i].Count > 0 then
      begin
        centroids[i].x := clusters[i].Average(p -> p.x);
        centroids[i].y := clusters[i].Average(p -> p.y);
      end;
  end;
  
  // Возвращаем массив точек в кластерах и центроиды
  Result := (clusters.Select(c -> c.ToArray).ToArray, centroids);
end;

begin
  // Сгенерируем исходные кластеры точек
  var cluster1 := GenerateCluster(5, 3, 2.5, 50);
  var cluster2 := GenerateCluster(7, -6, 1.5, 25);
  var cluster3 := GenerateCluster(-7, 2, 1, 44);
  
  var allPoints := cluster1 + cluster2 + cluster3;
  
  // Применяем алгоритм K-means для кластеризации
  var (clusters, centroids) := KMeans(allPoints, 3, 100);
  
  DrawPoints(clusters[0],3);
  DrawPoints(clusters[1],3);
  DrawPoints(clusters[2],PointRadius := 3);

  DrawPoint(centroids[0].x,centroids[0].y,PointRadius := 5);
  DrawPoint(centroids[1].x,centroids[1].y,PointRadius := 5);
  DrawPoint(centroids[2].x,centroids[2].y,PointRadius := 5);
  
  // Выводим кластеры и их центроиды
  (*for var i := 0 to clusters.Length - 1 do
  begin
    Writeln($'Cluster {i + 1}:');
    clusters[i].Println;
    Writeln($'Centroid {i + 1}: ({centroids[i].x}, {centroids[i].y})');
    Writeln;
  end;*)
end.