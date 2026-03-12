begin
  var a := ArrRandomInteger(10,2,5);
  a.Println;
  var count4 := a.CountOf(4); 
  var count5 := a.CountOf(5); 
  case count4.CompareTo(count5) of
    1: Print('четвёрок больше');
    0: Print('одинаково');
    -1: Print('пятёрок больше');
  end;

end.