type
  SortedMultiset<T> = class(IEnumerable<T>)
  private
    data := new SortedDictionary<T, integer>; // Храним элемент и количество его вхождений

  public
    // Метод добавления элемента
    procedure Add(x: T);
    begin
      if x in data then
        data[x] += 1 // Увеличиваем количество вхождений
      else data.Add(x, 1); // Добавляем новый элемент с количеством 1
    end;

    // Метод удаления элемента
    function Remove(x: T): boolean;
    begin
      if x in data then
      begin
        if data[x] > 1 then
          data[x] -= 1 // Уменьшаем количество вхождений
        else data.Remove(x); // Удаляем элемент, если это последнее вхождение
        Result := true;
      end
      else Result := false;
    end;

    // Метод получения количества вхождений элемента
    function Count(x: T): integer;
    begin
      if x in data then
        Result := data[x]
      else Result := 0;
    end;

    // Метод для получения минимального элемента
    function Min: T;
    begin
      if data.Count = 0 then
        raise new System.InvalidOperationException('Мультимножество пусто.');
      Result := data.Keys.First(); // Первый элемент в отсортированном множестве
    end;

    // Метод для получения максимального элемента
    function Max: T;
    begin
      if data.Count = 0 then
        raise new System.InvalidOperationException('Мультимножество пусто.');
      Result := data.Keys.Last(); // Последний элемент в отсортированном множестве
    end;

    // Метод для получения всех элементов множества
    function GetElements: sequence of T;
    begin
      foreach var key in data.Keys do
        for var i := 1 to data[key] do
          yield key;
    end;

    // Реализация интерфейса IEnumerable<T>
    function GetEnumerator: IEnumerator<T>;
    begin
      Result := GetElements().GetEnumerator();
    end;

    function System.Collections.IEnumerable.GetEnumerator: System.Collections.IEnumerator;
    begin
      Result := GetElements().GetEnumerator();
    end;
  end;

// Пример использования
begin
  var multiset := new SortedMultiset<integer>();
  multiset.Add(5);
  multiset.Add(3);
  multiset.Add(5);
  multiset.Add(2);
  
  multiset.Print(); // Вывод: 2 3 5 5
  
  multiset.Remove(5);
  multiset.Print(); // Вывод: 2 3 5
  
  Writeln('Min: ', multiset.Min); // Вывод: Min: 2
  Writeln('Max: ', multiset.Max); // Вывод: Max: 5
  Writeln('Количество 5: ', multiset.Count(5)); // Вывод: Количество 5: 1

  // Использование в цикле foreach
  foreach var elem in multiset do
    Writeln('Элемент: ', elem);
  
  multiset.Println;
end.