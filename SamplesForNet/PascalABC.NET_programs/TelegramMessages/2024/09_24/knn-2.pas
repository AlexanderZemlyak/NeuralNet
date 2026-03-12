type
  // Определяем автокласс для хранения данных о точке
  Point = auto class
    Features: array of Real; // Признаки точки
    &Label: String; // Метка класса, экранировано символом &
    
    // Конструктор с меткой
    public constructor(aFeatures: array of Real; aLabel: String);
    begin
      Features := aFeatures;
      &Label := aLabel; // Присваиваем значение метки
    end;
  end;

function EuclideanDistance(PointA, PointB: Point): Real;
begin
  Result := Sqrt((PointA.Features.Zip(PointB.Features, 
    (A, B) -> Sqr(A - B))).Sum);
end;

function KNN(Data: array of Point; Query: Point; K: Integer): String;
begin
  var Distances := Data.Select(P -> (EuclideanDistance(P, Query), P.&Label));

  var NearestNeighbors := Distances.OrderBy(D -> D.Item1).Take(K);

  // Используем EachCount для подсчета меток
  var CountLabels := NearestNeighbors.Select(N -> N.Item2).EachCount();

  Result := CountLabels.OrderByDescending(C -> C.Value).First().Key;
end;

// Основная часть программы
begin
  var TrainingData := | // Инициализация массива TrainingData с использованием ||
    new Point(|1.0, 2.0|, 'A'), 
    new Point(|1.5, 2.5|, 'A'), 
    new Point(|5.0, 4.0|, 'B'), 
    new Point(|5.5, 3.5|, 'B'), 
    new Point(|1.0, 4.0|, 'A') 
  |;

  var TestPoint := new Point(|1.2, 2.4|, ''); // Метка не известна
  var K := 3; // Количество соседей

  // Классификация тестовой точки
  var ResultLabel := KNN(TrainingData, TestPoint, K);
  Print('The predicted label for the test point is: ', ResultLabel);
end.