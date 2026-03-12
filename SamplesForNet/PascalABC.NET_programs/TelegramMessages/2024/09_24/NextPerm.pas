begin
  var a := Arr(1..6);
  loop 5 do
  begin
    NextPermutation(a);
    a.Println;
  end;
end.