import { BrowserRouter, Routes, Route, NavLink } from "react-router-dom";
import Dashboard from "./pages/Dashboard.tsx";
import Scan from "./pages/Scan.tsx";
import Training from "./pages/Training.tsx";
import Adversarial from "./pages/Adversarial.tsx";
import Pipeline from "./pages/Pipeline.tsx";
import Vault from "./pages/Vault.tsx";

const tabs = [
  { path: "/", label: "Dashboard", element: <Dashboard /> },
  { path: "/scan", label: "Scan", element: <Scan /> },
  { path: "/training", label: "Training", element: <Training /> },
  { path: "/adversarial", label: "Adversarial", element: <Adversarial /> },
  { path: "/pipeline", label: "Pipeline", element: <Pipeline /> },
  { path: "/vault", label: "Vault", element: <Vault /> },
];

export default function App() {
  return (
    <BrowserRouter>
      <div
        className="min-h-screen flex flex-col"
        style={{ backgroundColor: "var(--bg-primary)", color: "var(--text-primary)" }}
      >
        {/* Top nav bar */}
        <nav
          className="flex items-center gap-1 px-6 h-12 border-b"
          style={{ backgroundColor: "var(--bg-surface)", borderColor: "var(--border)" }}
        >
          <span
            className="text-sm font-bold tracking-wider mr-6"
            style={{ fontFamily: "var(--font-heading)", color: "var(--data)" }}
          >
            WINTERMUTE
          </span>
          {tabs.map((tab) => (
            <NavLink
              key={tab.path}
              to={tab.path}
              end={tab.path === "/"}
              className={({ isActive }) =>
                `px-3 py-2 text-xs tracking-wide transition-colors border-b-2 ${
                  isActive
                    ? "border-current"
                    : "border-transparent hover:border-current/30"
                }`
              }
              style={({ isActive }) => ({
                fontFamily: "var(--font-heading)",
                color: isActive ? "var(--data)" : "var(--text-muted)",
              })}
            >
              {tab.label.toUpperCase()}
            </NavLink>
          ))}
        </nav>

        {/* Page content */}
        <main className="flex-1 overflow-auto">
          <Routes>
            {tabs.map((tab) => (
              <Route key={tab.path} path={tab.path} element={tab.element} />
            ))}
          </Routes>
        </main>
      </div>
    </BrowserRouter>
  );
}
