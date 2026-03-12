// Функция для вычисления площади треугольника по координатам его вершин
function triangleArea(x1, y1, x2, y2, x3, y3: real): real;
begin
  Result := abs((x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)) / 2.0);
end;

// Функция проверки принадлежности точки треугольнику
function IsPointInTriangle(px, py, x1, y1, x2, y2, x3, y3: real): boolean;
begin
  // Площадь исходного треугольника
  var areaOriginal := triangleArea(x1, y1, x2, y2, x3, y3);
  
  // Площади треугольников, образованных с точкой
  var area1 := triangleArea(px, py, x2, y2, x3, y3);
  var area2 := triangleArea(x1, y1, px, py, x3, y3);
  var area3 := triangleArea(x1, y1, x2, y2, px, py);
  
  // Если сумма площадей равна исходной площади, точка принадлежит треугольнику
  Result := abs(areaOriginal - (area1 + area2 + area3)) < 1e-9;
end;

begin
  var (x1, y1) := (0.0, 0.0);  // Первая вершина треугольника
  var (x2, y2) := (5.0, 0.0);  // Вторая вершина треугольника
  var (x3, y3) := (0.0, 5.0);  // Третья вершина треугольника
  var (px, py) := (2.0, 2.0);  // Проверяемая точка

  // Проверка принадлежности точки треугольнику
  if IsPointInTriangle(px, py, x1, y1, x2, y2, x3, y3) then
    Println('Точка принадлежит треугольнику')
  else
    Println('Точка не принадлежит треугольнику');
end.