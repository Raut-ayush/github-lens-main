import { BarChart3, Code2, Sparkles, FolderGit2, ArrowLeft } from "lucide-react";
import { NavLink } from "@/components/NavLink";
import { Button } from "@/components/ui/button";
import { useNavigate } from "react-router-dom";
import {
  Sidebar,
  SidebarContent,
  SidebarGroup,
  SidebarGroupContent,
  SidebarGroupLabel,
  SidebarMenu,
  SidebarMenuButton,
  SidebarMenuItem,
  SidebarHeader,
} from "@/components/ui/sidebar";

interface AppSidebarProps {
  username: string;
}

export function AppSidebar({ username }: AppSidebarProps) {
  const navigate = useNavigate();
  
  const menuItems = [
    { title: "Overview", url: `/dashboard/${username}/overview`, icon: BarChart3 },
    { title: "Repositories", url: `/dashboard/${username}/repositories`, icon: FolderGit2 },
    { title: "Skills", url: `/dashboard/${username}/skills`, icon: Code2 },
    { title: "ML Insights", url: `/dashboard/${username}/ml-insights`, icon: Sparkles },
  ];
  return (
    <Sidebar collapsible="icon">
      <SidebarHeader className="border-b p-4">
        <Button
          variant="ghost"
          size="sm"
          onClick={() => navigate("/")}
          className="w-full justify-start"
        >
          <ArrowLeft className="h-4 w-4 mr-2" />
          <span>New Analysis</span>
        </Button>
      </SidebarHeader>
      <SidebarContent>
        <SidebarGroup>
          <SidebarGroupLabel>@{username}</SidebarGroupLabel>
          <SidebarGroupContent>
            <SidebarMenu>
              {menuItems.map((item) => (
                <SidebarMenuItem key={item.title}>
                  <SidebarMenuButton asChild>
                    <NavLink to={item.url}>
                      <item.icon />
                      <span>{item.title}</span>
                    </NavLink>
                  </SidebarMenuButton>
                </SidebarMenuItem>
              ))}
            </SidebarMenu>
          </SidebarGroupContent>
        </SidebarGroup>
      </SidebarContent>
    </Sidebar>
  );
}
