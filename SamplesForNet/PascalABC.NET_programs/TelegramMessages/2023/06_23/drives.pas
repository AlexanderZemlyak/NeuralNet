begin
  foreach var drive in System.IO.DriveInfo.GetDrives do
    Println(drive.Name,drive.DriveType,drive.VolumeLabel);
end.