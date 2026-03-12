function GetTopKMaximums(arr: array of integer; k: integer): array of integer;
begin
  // Минимальная куча (приоритетная очередь)
  var minHeap := new SortedSet<integer>(arr.Take(k));

  // Обрабатываем оставшиеся элементы массива
  for var i := k to arr.Length - 1 do
    if arr[i] > minHeap.Min then
    begin
      minHeap.Remove(minHeap.Min); // Удаляем минимальный элемент из кучи
      minHeap.Add(arr[i]); // Добавляем новый элемент
    end;

  // Преобразуем кучу в массив
  Result := arr.Where(x -> x in minHeap).OrderDescending.ToArray
end;

begin
  var arr := [3, 1, 4, 2, 5, 9, 7, 7, 6];
  var k := 4; // Количество максимумов
  var topKMaximums := GetTopKMaximums(arr, k);
  // Вывод результата
  Print($'Первые {k} максимума(ов):', topKMaximums);
end.