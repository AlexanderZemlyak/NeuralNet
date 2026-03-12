uses Accord;
uses Accord.MachineLearning;
uses Accord.DataSets;

begin
  var iris := new Iris();
  var inputs := iris.Instances;
  
  var km := new KMeans(3);
  
  var clusters := km.Learn(inputs);
  
  var labels := clusters.Decide(inputs);
  var groups := inputs.ZipTuple(labels).GroupBy(t -> t[1]).OrderBy(gr -> gr.Key);
  foreach var group in groups do
  begin
    Println(group.Key,'class  ');
    group.Take(5).Select(t->'  '+t[0].JoinToString).Concat(|'  ...'|).PrintLines
  end;
end.