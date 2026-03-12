// https://rosettacode.org/wiki/Write_to_Windows_event_log#PascalABC.NET

uses System.Diagnostics;

begin
  if not EventLog.SourceExists('MyApp') then
    EventLog.CreateEventSource('MyApp', 'Application');
  EventLog.WriteEntry('MyApp', 'Hello from PABC!');
end.