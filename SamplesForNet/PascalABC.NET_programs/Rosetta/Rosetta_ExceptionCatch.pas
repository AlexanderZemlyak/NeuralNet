// https://rosettacode.org/wiki/Exceptions/Catch_an_exception_thrown_in_a_nested_call#PascalABC.NET

type
  U0 = class(Exception) end;
  U1 = class(Exception) end;
  
var i := 0; 
  
procedure Baz;
begin
  if i = 0 then 
    raise new U0
  else raise new U1;
end;

procedure Bar;
begin
  Baz;
end;  
  
procedure Foo;
begin
  try
    bar;
  except
    on e: U0 do
      Println('U0 catched')
  end;
  i := 1;
  try
    bar;
  except
    on e: U0 do
      Println('U0 catched')
  end;
end;

begin
  Foo;
end.