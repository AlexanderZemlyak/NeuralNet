function MachineEpsilon: real;
begin
  var eps := 1.0;
  while 1.0 + eps > 1.0 do
    eps := eps / 2.0;
  Result := eps * 2.0;
end;

begin
  var eps := MachineEpsilon;
  Println('Машинный эпсилон =', eps);
  Println('1 + eps = 1?', 1 + eps = 1);
  Println('1 + eps/2 = 1?', 1 + eps/2 = 1);
end.