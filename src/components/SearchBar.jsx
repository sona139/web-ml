export function SearchBar({ search }) {
  return (
    <form
      onSubmit={(e) => {
        e.preventDefault();
        const formData = new FormData(e.target);
        const text = formData.get("text");
        search(text);
      }}
    >
      <input
        type="search"
        name="text"
        id="default-search"
        placeholder="Search for images..."
        required
      />
      <button type="submit">Search</button>
    </form>
  );
}
