import { Badge } from "@/components/ui/badge";

interface Skill {
  name: string;
  frequency?: number;
  percentage?: number;
}

interface SkillCloudProps {
  skills: Skill[];
}

export function SkillCloud({ skills }: SkillCloudProps) {
  const safeSkills = Array.isArray(skills) ? skills : [];

  const getSize = (percentage = 0) => {
    if (percentage >= 30) return "text-2xl px-4 py-2";
    if (percentage >= 20) return "text-xl px-3 py-2";
    if (percentage >= 10) return "text-lg px-3 py-1.5";
    return "text-base px-2 py-1";
  };

  const getVariant = (percentage = 0): "default" | "secondary" | "outline" => {
    if (percentage >= 20) return "default";
    if (percentage >= 10) return "secondary";
    return "outline";
  };

  return (
    <div className="flex flex-wrap gap-3 items-center justify-center p-8">
      {safeSkills.map((skill, index) => {
        const name = skill.name ?? "Unknown";
        const pct = skill.percentage ?? 0;
        return (
          <Badge
            key={`${name}-${index}`}
            variant={getVariant(pct)}
            className={`${getSize(pct)} transition-all hover:scale-110 cursor-default`}
          >
            {name}
          </Badge>
        );
      })}
    </div>
  );
}
