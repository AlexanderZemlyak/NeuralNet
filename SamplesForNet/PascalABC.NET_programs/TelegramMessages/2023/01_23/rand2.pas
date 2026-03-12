uses PlotWPF;

begin
  // Процент выпадения n орлов при 100 бросаниях
  var n := 1000000;
  var a := new real[101];
  loop n do
  begin
    var Орлы := 0;
    loop 100 do
      if Random(0, 1) = 0 then
        Орлы += 1;
    a[Орлы] += 1;
  end;
  LineGraphWPF.Create(PartitionPoints(0,100,100),a);
end.