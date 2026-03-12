begin
  var m := MatrRandom(3,4);
  m.Print;
  var m1 := m.ConvertAll((x,i,j) -> begin
    case i of
      0: Result := x + 100;
      1: Result := x + 500;
      2: Result := x + 300;
    end;
  end);
  Println;
  m1.Println;
end.