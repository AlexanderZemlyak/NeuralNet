type Person = auto class
  Name: string;
  Age: integer;
end;

begin
  var p := new Person('Иван',25);
  Println(p.Age);
  p := nil;
  Println(p?.Age);
  Println(p?.Age ?? 66);
end.