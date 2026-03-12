type
  Person = record
    name: string;
    age: integer;
  end;
  
function PersonCreate(name: string; age: integer): Person;
begin
  Result.name := name;
  Result.Age := age
end;

begin
  var p1 := PersonCreate('Иван',20);
  var p2 := PersonCreate('Иван',20);
  Print(p1 = p2);
end.  