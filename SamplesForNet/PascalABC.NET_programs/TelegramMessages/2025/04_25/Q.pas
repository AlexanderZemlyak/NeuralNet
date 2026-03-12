type
  Student = auto class
    name: string;
    age: integer;
  end;

begin
  var s: Student := nil;
  var name := s?.name;
  var age := s?.age ?? -1;
  Println(name,age);
  s := new Student('Иванов',18);
  name := s?.name;
  age := s?.age ?? -1;
  Println(name,age);
end.