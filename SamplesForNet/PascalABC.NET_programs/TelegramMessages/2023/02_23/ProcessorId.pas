{$reference System.Management.dll}
uses System.Management;

function ProcessorId: string;
begin
  var mbs := new ManagementObjectSearcher('Select ProcessorId From Win32_processor');
  var mbsList := mbs.Get();
  Result := '';
  foreach var mo: ManagementObject in mbsList do
  begin
    Result := mo['ProcessorId'].ToString();
    break;
  end;
end;

begin
  Print(ProcessorId);
end.