uses Coords;

function GenerateCluster(X, Y, spread: real; count: integer): array of Point;
begin
  Result := ArrGen(count, i -> new Point(
    RandomReal(X - spread / 2, X + spread / 2),
    RandomReal(Y - spread / 2, Y + spread / 2)
  ));
end;

begin
  Window.Title := 'Генерация кластеров';
  var cluster1 := GenerateCluster(5, 3, 5, 90);
  var cluster2 := GenerateCluster(7, -6, 4, 105);
  var cluster3 := GenerateCluster(-7, 2, 6, 44);
  
  DrawPoints(cluster1,3);
  DrawPoints(cluster2,3);
  DrawPoints(cluster3,PointRadius := 3);
end.