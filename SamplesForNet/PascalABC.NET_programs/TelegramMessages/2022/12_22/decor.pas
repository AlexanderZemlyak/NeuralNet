function pdecorated(p: procedure): procedure := procedure -> begin
    Println('Before');
    p;
    Println('After');
  end;

begin
  var p := procedure -> Println('p');
  p;
  p := pdecorated(p);
  p;
end.

