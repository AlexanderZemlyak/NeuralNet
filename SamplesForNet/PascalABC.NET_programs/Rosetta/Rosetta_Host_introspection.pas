// https://rosettacode.org/wiki/Host_introspection#PascalABC.NET

begin
  Println($'Word size in bytes: {sizeof(integer)}');
  if System.BitConverter.IsLittleEndian then
    Println('Little Endian')
  else Println('Big Endian')
end.