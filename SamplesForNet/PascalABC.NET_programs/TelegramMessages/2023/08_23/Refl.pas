begin
  foreach var mi in typeof(string).GetMethods() do
    Writeln($'function ',mi.Name,
      '('+mi.GetParameters.Select(pi -> TypeToTypeName(pi.ParameterType)).JoinToString(', '),')',
      ': ',
      TypeToTypeName(mi.ReturnType))
end.