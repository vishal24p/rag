export const documents = [
  {
    id: "strategy",
    name: "Q4 Product Strategy.pdf",
    type: "PDF",
    size: "2.4 MB",
    pages: 18,
    color: "rose",
    excerpt:
      "Our Q4 focus is a smaller, more dependable release. The launch should prioritize the core onboarding flow, reliable document retrieval, and clear source attribution. Advanced integrations follow after the core experience is validated.",
    page: 4,
  },
  {
    id: "research",
    name: "Customer Research.md",
    type: "MD",
    size: "128 KB",
    pages: 8,
    color: "blue",
    excerpt:
      "Customers want to reach a useful answer quickly and understand where it came from. The strongest recurring requests were a shorter onboarding path, trustworthy citations, and a simple way to explore the source material.",
    page: 2,
  },
  {
    id: "roadmap",
    name: "Launch Roadmap.pdf",
    type: "PDF",
    size: "1.8 MB",
    pages: 12,
    color: "amber",
    excerpt:
      "Launch dependencies: complete retrieval quality checks, confirm the onboarding handoff, and resolve source permission behavior. Assign owners to each dependency and review readiness before announcing a release date.",
    page: 6,
  },
];
export const openingQuestion =
  "What should we focus on for the Q4 product launch?";
export const openingAnswer = `The documents point to one clear direction: **make the core experience feel effortless before expanding it.**

### Three priorities for the launch

1. **Get people to their first useful answer faster.** Simplify onboarding and make the next step obvious. This is the strongest theme in the customer research.
2. **Build trust into every answer.** Prioritize reliable retrieval, clear citations, and easy access to the original document.
3. **Keep the release focused.** Ship the essential workflow first; move advanced integrations into a follow-up release.

The strategy and roadmap agree on the scope. The open question is readiness: **who owns each remaining dependency, and when can it be validated?**`;
export function validPrompt(text: string) {
  return text.trim().length > 0;
}
// ponytail: local sample answers; replace with your API transport when backend integration is requested.
export function createDemoAnswer(prompt: string) {
  if (/risk|block|depend|ready/i.test(prompt))
    return `### What could hold up the launch?

The sample roadmap identifies three **dependencies** to resolve:

- **Retrieval quality:** validate that answers cite the right passages.
- **Onboarding handoff:** confirm ownership of the first-use experience.
- **Source permissions:** verify people can only access permitted documents.

Assign an owner and a readiness check to each item before setting the release date. These are demo document excerpts, not a live assessment.`;
  if (/compare|research|customer/i.test(prompt))
    return `### Where strategy and research meet

The sample strategy favors a focused release. Customer research supports that choice: people want **fast onboarding, trustworthy answers, and visible sources**.

The practical next step is to validate that complete workflow with users before adding integrations. The roadmap lists the remaining dependencies, but does not establish a confirmed launch date.`;
  return (
    openingAnswer +
    "\n\n*This is a sample response from the demo documents. Connect your RAG API to answer arbitrary questions.*"
  );
}
