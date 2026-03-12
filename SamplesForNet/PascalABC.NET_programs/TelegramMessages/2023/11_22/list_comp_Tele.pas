// https://en.wikipedia.org/wiki/Comparison_of_programming_languages_(list_comprehension)
begin
  // Возвращается массив
  var d := [s for s in stud where s.Age>20 orderby s.Name];
  // Возвращается последовательност
  var seq := s for s in stud where s.Age>20 orderby s.Name;
  // т.е. последовательность берешь в [] и получаешь массив - если справлюсь с грамматикой
  var a := [a.Where(x->x>0)];
  
  // Варианты синтаксиса генераторов списков
  
  // Альтернатива 0 - LINQ C#
  var a := for x in a where x>0 select x*x;
  // Альтернатива 1 - почти как в LINQ
  var a := [x*x from x in a where x>0];
  // Альтернатива 2 - если не указывается ЧТО, то берется выражение после for
  var a := [for x in a where x>0];
  // Альтернатива 3 - чуть математическая нотация
  var a := [x*x: for x in 1..10 where x.IsOdd];
  // Альтернатива 4 - Nemerle, математическая нотация, но тогда orderby непонятно как прицепить
  var a := [x*x: x in 1..10, x.IsOdd];
  // Альтернатива 5 - Julia и Python - слово if
  // вложенные for разумеется можно делать везде
  var a := [x*y for x in 1..5 for y in 1..5 if x.IsOdd and y.IsOdd];
end.