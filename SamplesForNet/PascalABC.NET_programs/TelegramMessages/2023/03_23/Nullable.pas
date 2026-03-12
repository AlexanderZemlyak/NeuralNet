type 
  Person = auto class
    Name: string;
    Age: integer;
  end;
  
begin
  var p: Person;
  Print(p?.Name);
  Print(p?.Age ?? 17);
end.  