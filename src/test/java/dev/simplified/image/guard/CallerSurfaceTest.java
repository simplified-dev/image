package dev.simplified.image.guard;

import dev.simplified.image.codec.ImageReader;
import dev.simplified.image.codec.ImageWriter;
import dev.simplified.image.option.ImageReadOptions;
import dev.simplified.image.option.ImageWriteOptions;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.lang.reflect.Constructor;
import java.lang.reflect.Executable;
import java.lang.reflect.Field;
import java.lang.reflect.GenericArrayType;
import java.lang.reflect.Method;
import java.lang.reflect.Modifier;
import java.lang.reflect.ParameterizedType;
import java.lang.reflect.RecordComponent;
import java.lang.reflect.Type;
import java.lang.reflect.TypeVariable;
import java.lang.reflect.WildcardType;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.Set;
import java.util.TreeSet;
import java.util.function.BiConsumer;
import java.util.function.Consumer;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.stream.Stream;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.empty;
import static org.hamcrest.Matchers.greaterThan;
import static org.hamcrest.Matchers.is;

/**
 * The caller surface, held against the library's own main sources.
 * <p>
 * Every package is a caller package or an internal one, decided by name. A caller imports from the root,
 * {@code frame}, {@code pixel}, {@code option}, {@code exception} and {@code codec}; every package below
 * {@code codec} is internal - a format's reader and writer, the WebP bitstreams, and the helpers the
 * formats share. Six rules keep that line where a caller meets it:
 * <ul>
 *   <li><b>Charters.</b> Every package is one or the other, and carries a {@code package-info.java} whose
 *   doc comment says what belongs in it, with no imports.</li>
 *   <li><b>Imports.</b> No caller source imports an internal type, except a gateway on {@link #GATEWAYS} -
 *   {@code ImageFactory}, the one route to a codec.</li>
 *   <li><b>Signatures.</b> No public or protected signature of a caller type names an internal type,
 *   however it is spelled and whether it is written or generated.</li>
 *   <li><b>Kinds.</b> Every options type lives in {@code option}, and every concrete reader and writer in
 *   a format's package one level below {@code codec}.</li>
 *   <li><b>Shared helpers.</b> {@code codec.common} and every package below it name no format, so no
 *   format depends on another through them.</li>
 *   <li><b>Public internals.</b> Every type in an internal package that another package can reach carries
 *   {@code @ApiStatus.Internal} and is named from another package. Reach is the type's effective
 *   visibility: it is public, or nested and protected, and so is every type it is nested in, where a type
 *   declared inside an interface is public whether or not it says so. One that no other package names
 *   should not be reachable.</li>
 * </ul>
 * <p>
 * An import counts when the file's code uses it: comments, text blocks and string and character literals
 * are stripped before the imports are read and the imported simple name is looked for, so a javadoc-only
 * import is not an edge and a trailing comment hides none. An on-demand import of an internal package
 * always counts, because the text cannot say which of the names it supplies the code uses.
 * Test sources sit outside every rule, because the white-box tests reach package-private members by
 * design.
 * <p>
 * {@link #KNOWN} is the ledger of import edges that break the import or shared-helper rule today, each
 * with why it stands. It is held exactly: an edge that breaks a rule and is not on it fails, and an entry
 * the tree no longer carries fails too, so the ledger only ever shrinks.
 * <p>
 * Nothing here holds the codec set closed. A static {@code registerReader} put back on
 * {@code ImageFactory} would pass every rule, because {@code codec} is a caller package; review holds
 * that.
 */
@DisplayName("A caller reaches the codecs only through ImageFactory")
class CallerSurfaceTest {

    /** The library's root package, which every package below is named relative to. */
    private static final String BASE = "dev.simplified.image";

    /** The main source root, relative to the project directory Gradle runs tests from. */
    private static final Path MAIN = Path.of("src/main/java");

    /** The packages a caller imports by hand, relative to the root, the root itself spelled empty. */
    private static final Set<String> CALLER = Set.of("", "frame", "pixel", "option", "exception", "codec");

    /** The prefix every internal package carries, relative to the root. */
    private static final String INTERNAL = "codec.";

    /** The package the codecs share helpers through, which names no format, and nor does any package below it. */
    private static final String COMMON = "codec.common";

    /** The package every options type lives in. */
    private static final String OPTION = "option";

    /** The caller types that may import an internal package, each relative to the root, with why. */
    private static final Map<String, String> GATEWAYS = Map.of(
        "ImageFactory", "its static initializer constructs every built-in reader and writer");

    /** The import edges that break the import or shared-helper rule today, each with why it stands. */
    private static final Map<String, String> KNOWN = Map.of();

    /** A package declaration. */
    private static final Pattern PACKAGE = Pattern.compile("(?m)^package\\s+([\\w.]+)\\s*;");

    /** An import, static or not, of one name or on demand, read from a source's stripped text. */
    private static final Pattern IMPORT = Pattern.compile("(?m)^import\\s+(static\\s+)?([\\w.]+(?:\\.\\*)?)\\s*;[ \\t]*$");

    /** Comments, text blocks, string and character literals - text a type name can sit in without being used. */
    private static final Pattern NOISE = Pattern.compile(
        "/\\*.*?\\*/|//[^\\n]*|\"\"\".*?\"\"\"|\"(?:\\\\.|[^\"\\\\])*\"|'(?:\\\\.|[^'\\\\])*'", Pattern.DOTALL);

    /** The internal tag, by its imported spelling or its fully qualified one. */
    private static final Pattern TAG = Pattern.compile("@(?:org\\.jetbrains\\.annotations\\.)?ApiStatus\\.Internal\\b");

    /** A type declaration: its annotations and modifiers, its kind and its name. */
    private static final Pattern DECLARATION = Pattern.compile(
        "(?<![\\w.$@])((?:(?:@[\\w.]+(?:\\s*\\((?:[^()]|\\([^()]*\\))*\\))?|public|protected|private|static"
            + "|final|abstract|sealed|non-sealed|strictfp)\\s+)*)(class|interface|enum|record|@interface)\\s+(\\w+)");

    /**
     * One main source file.
     *
     * @param path where it is
     * @param pkg its declared package
     * @param stripped its text with comments and literals blanked, imports, annotations and declarations intact
     * @param code its stripped text with the import lines blanked too, which is what a use is looked for in
     */
    private record Source(Path path, String pkg, String stripped, String code) {}

    /**
     * One type declared in an internal source.
     *
     * @param chain its simple name, preceded by the simple name of every type it is nested in
     * @param isTagged whether the declaration carries the internal tag
     */
    private record Declaration(List<String> chain, boolean isTagged) {}

    @Test
    @DisplayName("every package is a caller or an internal one, and carries a charter")
    void everyPackageIsClassifiedAndChartered() {
        Set<String> unclassified = new TreeSet<>();
        Set<String> unchartered = new TreeSet<>();
        for (String pkg : declaredPackages()) {
            Optional<String> name = relative(pkg);
            if (name.isEmpty() || !(isCaller(name.get()) || isInternal(name.get()))) unclassified.add(pkg);
            if (!isChartered(MAIN.resolve(pkg.replace('.', '/')).resolve("package-info.java"))) unchartered.add(pkg);
        }

        assertThat("packages that are neither a caller package nor below codec. A new package is classified "
            + "deliberately, in CALLER or under codec, rather than by where it happens to sit", unclassified, is(empty()));
        assertThat("packages with no charter: a package-info.java whose doc comment comes ahead of the package "
            + "line, and which imports nothing - a charter links by fully qualified name", unchartered, is(empty()));
    }

    @Test
    @DisplayName("no caller source imports an internal type, except a gateway")
    void callerSourcesImportNoInternals() {
        List<Edge> edges = callerEdges();
        List<String> unrecorded = callerViolations(edges).stream()
            .filter(edge -> !KNOWN.containsKey(edge.key()))
            .map(edge -> edge.key() + " (" + edge.path() + ")")
            .toList();
        List<String> idle = GATEWAYS.keySet().stream()
            .filter(gateway -> edges.stream().noneMatch(edge -> edge.from().equals(gateway)))
            .sorted()
            .toList();

        assertThat("caller sources importing an internal type. Reach the codec through ImageFactory, or move the "
            + "type that needs it below codec, rather than recording it", unrecorded, is(empty()));
        assertThat("gateways that import no internal type, and so need no exemption", idle, is(empty()));
    }

    @Test
    @DisplayName("codec.common names no format")
    void commonNamesNoFormat() {
        List<String> unrecorded = commonEdges().stream()
            .filter(edge -> !KNOWN.containsKey(edge.key()))
            .map(edge -> edge.key() + " (" + edge.path() + ")")
            .toList();

        assertThat("sources in or below codec.common importing a type of another codec package. A shared helper names no "
            + "format, so that no format comes to depend on another through it", unrecorded, is(empty()));
    }

    @Test
    @DisplayName("the ledger of known edges is exact")
    void knownLedgerIsExact() {
        Set<String> carried = new TreeSet<>();
        callerViolations(callerEdges()).forEach(edge -> carried.add(edge.key()));
        commonEdges().forEach(edge -> carried.add(edge.key()));
        List<String> paid = KNOWN.keySet().stream().filter(edge -> !carried.contains(edge)).sorted().toList();

        assertThat("ledger entries the tree no longer carries. The debt is paid, so the entry goes", paid, is(empty()));
    }

    /**
     * One import a source's code uses, from the type the file declares to the type it imports.
     *
     * @param from the importing file's type, relative to the root
     * @param to the imported type, relative to the root
     * @param path the importing file
     */
    private record Edge(String from, String to, Path path) {

        /**
         * Spells the edge the way {@link #KNOWN} keys it.
         *
         * @return {@code from -> to}
         */
        String key() {
            return this.from + " -> " + this.to;
        }

    }

    /**
     * Collects every import by which a caller source's code names an internal type.
     *
     * @return the edges, gateways' included
     */
    private static List<Edge> callerEdges() {
        List<Edge> edges = new ArrayList<>();
        for (Source source : sources()) {
            Optional<String> from = relative(source.pkg());
            if (from.isEmpty() || !isCaller(from.get()) || isCharter(source)) continue;
            for (String imported : usedImports(source)) {
                Optional<String> to = packageOf(imported).flatMap(CallerSurfaceTest::relative);
                if (to.isPresent() && isInternal(to.get()))
                    edges.add(new Edge(typeName(source), imported.substring(BASE.length() + 1), source.path()));
            }
        }
        return edges;
    }

    /**
     * Drops the edges a gateway accounts for.
     *
     * @param edges every caller-to-internal edge
     * @return the edges whose importing type is not a gateway
     */
    private static List<Edge> callerViolations(List<Edge> edges) {
        return edges.stream().filter(edge -> !GATEWAYS.containsKey(edge.from())).toList();
    }

    /**
     * Collects every import by which the code of a source in {@code codec.common}, or in a package below it,
     * names a type of another internal package.
     *
     * @return the edges
     */
    private static List<Edge> commonEdges() {
        List<Edge> edges = new ArrayList<>();
        for (Source source : sources()) {
            if (relative(source.pkg()).filter(CallerSurfaceTest::isCommon).isEmpty() || isCharter(source)) continue;
            for (String imported : usedImports(source)) {
                Optional<String> to = packageOf(imported).flatMap(CallerSurfaceTest::relative);
                if (to.isPresent() && isInternal(to.get()) && !isCommon(to.get()))
                    edges.add(new Edge(typeName(source), imported.substring(BASE.length() + 1), source.path()));
            }
        }
        return edges;
    }

    @Test
    @DisplayName("no caller signature names an internal type")
    void callerSignaturesNameNoInternals() {
        List<Class<?>> exposed = new ArrayList<>();
        for (Source source : sources()) {
            Optional<String> name = relative(source.pkg());
            if (name.isEmpty() || !isCaller(name.get()) || isCharter(source)) continue;
            Class<?> type = load(source.pkg() + "." + stem(source.path()));
            if (Modifier.isPublic(type.getModifiers())) expose(type, exposed);
        }

        Set<String> violations = new TreeSet<>();
        for (Class<?> type : exposed) {
            signature(type, (where, part) -> reach(part, new HashSet<>(), named -> {
                if (relative(named.getPackageName()).filter(CallerSurfaceTest::isInternal).isPresent())
                    violations.add(type.getName() + " " + where + " -> " + named.getName());
            }));
        }

        assertThat("no public caller type was read, so the rule held nothing", exposed.size(), is(greaterThan(0)));
        assertThat("caller signatures naming an internal type. A caller would then have to name it too; take or "
            + "answer a caller type instead", violations, is(empty()));
    }

    /**
     * Adds a type a caller can name, and every public or protected type nested in it, to a list.
     *
     * @param type the type, already known to be reachable
     * @param out the list to add to
     */
    private static void expose(Class<?> type, List<Class<?>> out) {
        out.add(type);
        for (Class<?> nested : type.getDeclaredClasses()) {
            int modifiers = nested.getModifiers();
            if (Modifier.isPublic(modifiers) || Modifier.isProtected(modifiers)) expose(nested, out);
        }
    }

    /**
     * Hands every type a caller-visible signature of one type is made of to a visitor: its supertypes and
     * type parameters, its record components, and its public and protected constructors, methods and
     * fields, generated members included.
     *
     * @param type the type
     * @param visitor what receives each part, with where in the signature it was found
     */
    private static void signature(Class<?> type, BiConsumer<String, Type> visitor) {
        if (type.getGenericSuperclass() != null) visitor.accept("extends", type.getGenericSuperclass());
        for (Type parent : type.getGenericInterfaces()) visitor.accept("implements", parent);
        for (TypeVariable<?> variable : type.getTypeParameters()) visitor.accept("<" + variable + ">", variable);
        if (type.isRecord()) {
            for (RecordComponent component : type.getRecordComponents())
                visitor.accept("component " + component.getName(), component.getGenericType());
        }
        for (Constructor<?> constructor : type.getDeclaredConstructors())
            if (isVisible(constructor.getModifiers())) executable("constructor", constructor, visitor);
        for (Method method : type.getDeclaredMethods()) {
            if (!isVisible(method.getModifiers())) continue;
            visitor.accept(method.getName() + " returns", method.getGenericReturnType());
            executable(method.getName(), method, visitor);
        }
        for (Field field : type.getDeclaredFields())
            if (isVisible(field.getModifiers())) visitor.accept("field " + field.getName(), field.getGenericType());
    }

    /**
     * Hands the parameters, thrown types and type parameters of a constructor or method to a visitor.
     *
     * @param where the member's name, for the report
     * @param executable the constructor or method
     * @param visitor what receives each part
     */
    private static void executable(String where, Executable executable, BiConsumer<String, Type> visitor) {
        for (Type parameter : executable.getGenericParameterTypes()) visitor.accept(where + " takes", parameter);
        for (Type thrown : executable.getGenericExceptionTypes()) visitor.accept(where + " throws", thrown);
        for (TypeVariable<?> variable : executable.getTypeParameters()) visitor.accept(where + " <" + variable + ">", variable);
    }

    /**
     * Walks a generic type down to every class it names - type arguments, wildcard and type-variable bounds
     * and array components included.
     *
     * @param type the type to walk
     * @param seen the types already walked, which stops a recursive bound such as {@code T extends Comparable<T>}
     * @param sink what receives each class
     */
    private static void reach(Type type, Set<Type> seen, Consumer<Class<?>> sink) {
        if (type == null || !seen.add(type)) return;
        switch (type) {
            case Class<?> clazz when clazz.isArray() -> reach(clazz.getComponentType(), seen, sink);
            case Class<?> clazz -> sink.accept(clazz);
            case ParameterizedType parameterized -> {
                reach(parameterized.getRawType(), seen, sink);
                reach(parameterized.getOwnerType(), seen, sink);
                for (Type argument : parameterized.getActualTypeArguments()) reach(argument, seen, sink);
            }
            case WildcardType wildcard -> {
                for (Type bound : wildcard.getUpperBounds()) reach(bound, seen, sink);
                for (Type bound : wildcard.getLowerBounds()) reach(bound, seen, sink);
            }
            case TypeVariable<?> variable -> {
                for (Type bound : variable.getBounds()) reach(bound, seen, sink);
            }
            case GenericArrayType array -> reach(array.getGenericComponentType(), seen, sink);
            default -> { }
        }
    }

    /**
     * Answers whether a member is part of what a caller sees.
     *
     * @param modifiers the member's modifiers
     * @return {@code true} for a public or protected member
     */
    private static boolean isVisible(int modifiers) {
        return Modifier.isPublic(modifiers) || Modifier.isProtected(modifiers);
    }

    @Test
    @DisplayName("every options type lives in option, and every reader and writer in a format's package")
    void eachKindLivesInItsPackage() {
        Set<String> misplaced = new TreeSet<>();
        int options = 0;
        int codecs = 0;
        for (Source source : sources()) {
            if (isCharter(source)) continue;
            List<Class<?>> types = new ArrayList<>();
            nest(load(source.pkg() + "." + stem(source.path())), types);
            for (Class<?> type : types) {
                String pkg = relative(type.getPackageName()).orElse(type.getPackageName());
                if (ImageReadOptions.class.isAssignableFrom(type) || ImageWriteOptions.class.isAssignableFrom(type)) {
                    options++;
                    if (!pkg.equals(OPTION)) misplaced.add(type.getName() + " configures a read or a write outside " + OPTION);
                }
                boolean concrete = !type.isInterface() && !Modifier.isAbstract(type.getModifiers());
                if (concrete && (ImageReader.class.isAssignableFrom(type) || ImageWriter.class.isAssignableFrom(type))) {
                    codecs++;
                    if (!isFormatPackage(pkg)) misplaced.add(type.getName() + " reads or writes outside a format's package");
                }
            }
        }

        assertThat("no options type was read, so the rule held nothing", options, is(greaterThan(0)));
        assertThat("no reader or writer was read, so the rule held nothing", codecs, is(greaterThan(0)));
        assertThat("types outside the package their kind lives in. Options live in option, so a caller configures "
            + "a write without naming a codec, and each format's reader and writer one level below codec, never in "
            + COMMON, misplaced, is(empty()));
    }

    /**
     * Adds a type and every type nested in it, whatever its visibility, to a list.
     *
     * @param type the type
     * @param out the list to add to
     */
    private static void nest(Class<?> type, List<Class<?>> out) {
        out.add(type);
        for (Class<?> nested : type.getDeclaredClasses()) nest(nested, out);
    }

    /**
     * Answers whether a package is one format's codec package.
     *
     * @param name the package, relative to the root
     * @return {@code true} for a package exactly one level below {@code codec}, other than {@link #COMMON}
     */
    private static boolean isFormatPackage(String name) {
        return isInternal(name) && name.indexOf('.', INTERNAL.length()) < 0 && !name.equals(COMMON);
    }

    @Test
    @DisplayName("every internal type another package can reach is tagged internal and named from another package")
    void publicInternalsAreTaggedAndNamed() {
        Set<String> untagged = new TreeSet<>();
        Set<String> unnamed = new TreeSet<>();
        int found = 0;
        for (Source source : sources()) {
            if (relative(source.pkg()).filter(CallerSurfaceTest::isInternal).isEmpty() || isCharter(source)) continue;
            Map<List<String>, Boolean> tags = tags(source);
            List<Class<?>> types = new ArrayList<>();
            nest(load(source.pkg() + "." + stem(source.path())), types);
            for (Class<?> type : types) {
                if (!isReachable(type)) continue;
                found++;
                List<String> chain = chain(type);
                String name = source.pkg() + "." + String.join(".", chain);
                Boolean tagged = tags.get(chain);
                if (tagged == null)
                    throw new IllegalStateException("No declaration of '" + name + "' was read from '" + source.path() + "'");
                if (!tagged) untagged.add(name);
                if (!isNamedElsewhere(source.pkg(), chain)) unnamed.add(name);
            }
        }

        assertThat("no reachable internal type was read, so the rule held nothing", found, is(greaterThan(0)));
        assertThat("internal types another package can reach without @ApiStatus.Internal. Java has no sub-package "
            + "access, so a type another package calls stays public, and the tag is what tells a caller to keep out. "
            + "A type declared inside an interface is public whether or not it says so", untagged, is(empty()));
        assertThat("internal types another package can reach that no other main package names. Nothing needs them "
            + "reachable, so they should be package-private, or nested in a type that is", unnamed, is(empty()));
    }

    /**
     * Answers whether code in another package can name a type, by its effective visibility: it is public, or
     * nested and protected, and so is every type it is nested in. The compiler records a type declared inside
     * an interface as public whether or not it says so, and reflection reads that record.
     *
     * @param type the type
     * @return {@code true} when every link from it out to its top-level type is reachable
     */
    private static boolean isReachable(Class<?> type) {
        for (Class<?> link = type; link != null; link = link.getEnclosingClass()) {
            int modifiers = link.getModifiers();
            boolean nested = link.getEnclosingClass() != null;
            if (!Modifier.isPublic(modifiers) && !(nested && Modifier.isProtected(modifiers))) return false;
        }
        return true;
    }

    /**
     * Names a type by its simple name, preceded by the simple name of every type it is nested in.
     *
     * @param type the type
     * @return the chain, outermost first
     */
    private static List<String> chain(Class<?> type) {
        List<String> chain = new ArrayList<>();
        for (Class<?> link = type; link != null; link = link.getEnclosingClass()) chain.addFirst(link.getSimpleName());
        return List.copyOf(chain);
    }

    /**
     * Reads which of a source's type declarations carry the internal tag, which reflection cannot see: the tag
     * is kept in the class file but not at runtime. Where two declarations share a chain - a member and a local
     * class of one name - the chain counts as tagged only when both are, so the rule errs toward failing.
     *
     * @param source the source
     * @return whether each declared chain is tagged
     */
    private static Map<List<String>, Boolean> tags(Source source) {
        Map<List<String>, Boolean> tags = new HashMap<>();
        for (Declaration declaration : declarations(source))
            tags.merge(declaration.chain(), declaration.isTagged(), Boolean::logicalAnd);
        return tags;
    }

    /**
     * Reads every type declaration of a source, with the types it is nested in.
     *
     * @param source the source
     * @return its declarations, in the order they are written
     */
    private static List<Declaration> declarations(Source source) {
        String text = source.stripped();
        List<int[]> bodies = new ArrayList<>();
        List<String> names = new ArrayList<>();
        List<Declaration> out = new ArrayList<>();
        Matcher matcher = DECLARATION.matcher(text);
        while (matcher.find()) {
            int open = text.indexOf('{', matcher.end());
            int close = open < 0 ? text.length() : closing(text, open);
            List<String> chain = new ArrayList<>();
            for (int i = 0; i < bodies.size(); i++)
                if (bodies.get(i)[0] < matcher.start() && matcher.start() < bodies.get(i)[1]) chain.add(names.get(i));
            chain.add(matcher.group(3));
            bodies.add(new int[] { open, close });
            names.add(matcher.group(3));
            out.add(new Declaration(List.copyOf(chain), TAG.matcher(matcher.group(1)).find()));
        }
        return out;
    }

    /**
     * Finds the brace that closes a body.
     *
     * @param text the stripped text, so no brace sits in a comment or a literal
     * @param open the index of the body's opening brace
     * @return the index of its closing brace, or the end of the text when it has none
     */
    private static int closing(String text, int open) {
        int depth = 0;
        for (int i = open; i < text.length(); i++) {
            if (text.charAt(i) == '{') depth++;
            else if (text.charAt(i) == '}' && --depth == 0) return i;
        }
        return text.length();
    }

    /**
     * Answers whether a main source outside a type's package names it in code - by importing it, or by
     * importing a type it is nested in and writing the rest of the chain through that type. An on-demand
     * import imports every type declared directly in the package or type it names.
     *
     * @param pkg the type's package
     * @param chain the type's simple name, preceded by those of the types it is nested in
     * @return {@code true} when another package's code names the type
     */
    private static boolean isNamedElsewhere(String pkg, List<String> chain) {
        for (Source source : sources()) {
            if (source.pkg().equals(pkg)) continue;
            Matcher imports = IMPORT.matcher(source.stripped());
            while (imports.find()) {
                String imported = imports.group(2);
                boolean onDemand = imported.endsWith(".*");
                for (int i = 0; i < chain.size(); i++) {
                    String head = pkg + "." + String.join(".", chain.subList(0, i + 1));
                    if (!onDemand && imports.group(1) != null && imported.startsWith(head + ".") && uses(source, last(imported)))
                        return true;
                    String reached = onDemand ? imported.substring(0, imported.length() - 1) + chain.get(i) : imported;
                    if (!reached.equals(head)) continue;
                    String spelled = String.join("\\s*\\.\\s*", chain.subList(i, chain.size()).stream().map(Pattern::quote).toList());
                    if (Pattern.compile("\\b" + spelled + "\\b").matcher(source.code()).find()) return true;
                }
            }
        }
        return false;
    }

    /**
     * Names a package relative to the library root.
     *
     * @param pkg the fully qualified package
     * @return the relative name, empty for the root itself, or no name for a package outside the root
     */
    private static Optional<String> relative(String pkg) {
        if (pkg.equals(BASE)) return Optional.of("");
        if (pkg.startsWith(BASE + ".")) return Optional.of(pkg.substring(BASE.length() + 1));
        return Optional.empty();
    }

    /**
     * Answers whether a package is one a caller imports by hand.
     *
     * @param name the package, relative to the root
     * @return {@code true} for a package in {@link #CALLER}
     */
    private static boolean isCaller(String name) {
        return CALLER.contains(name);
    }

    /**
     * Answers whether a package is internal.
     *
     * @param name the package, relative to the root
     * @return {@code true} for every package below {@code codec}
     */
    private static boolean isInternal(String name) {
        return name.startsWith(INTERNAL);
    }

    /**
     * Answers whether a package is the one the codecs share helpers through, or one below it.
     *
     * @param name the package, relative to the root
     * @return {@code true} for {@link #COMMON} and every package below it
     */
    private static boolean isCommon(String name) {
        return name.equals(COMMON) || name.startsWith(COMMON + ".");
    }

    /**
     * Resolves the package an imported name belongs to, walking nested types and static members back to it.
     *
     * @param imported the imported name
     * @return the longest declared main package that prefixes it, or empty for a name outside the library
     */
    private static Optional<String> packageOf(String imported) {
        String candidate = imported;
        while (candidate.contains(".")) {
            candidate = candidate.substring(0, candidate.lastIndexOf('.'));
            if (declaredPackages().contains(candidate)) return Optional.of(candidate);
        }
        return Optional.empty();
    }

    /**
     * Lists the imports of a source its code uses, counting every on-demand import as used.
     *
     * @param source the source
     * @return each on-demand import, and each other imported name whose last segment the code names
     */
    private static List<String> usedImports(Source source) {
        List<String> used = new ArrayList<>();
        Matcher imports = IMPORT.matcher(source.stripped());
        while (imports.find()) {
            String imported = imports.group(2);
            if (imported.endsWith(".*") || uses(source, last(imported))) used.add(imported);
        }
        return used;
    }

    /**
     * Answers whether a source's code names a simple name.
     *
     * @param source the source
     * @param simple the simple name
     * @return {@code true} when the name appears outside comments, literals and import lines
     */
    private static boolean uses(Source source, String simple) {
        return Pattern.compile("\\b" + Pattern.quote(simple) + "\\b").matcher(source.code()).find();
    }

    /**
     * Returns the last segment of a dotted name.
     *
     * @param dotted the name
     * @return what follows its last dot
     */
    private static String last(String dotted) {
        return dotted.substring(dotted.lastIndexOf('.') + 1);
    }

    /**
     * Names the top-level type a source declares, relative to the root.
     *
     * @param source the source
     * @return its package, relative to the root, joined to its file name
     */
    private static String typeName(Source source) {
        String pkg = relative(source.pkg()).orElse(source.pkg());
        return pkg.isEmpty() ? stem(source.path()) : pkg + "." + stem(source.path());
    }

    /**
     * Returns a source file's name without its extension.
     *
     * @param path the file
     * @return the name of the top-level type it declares
     */
    private static String stem(Path path) {
        String name = path.getFileName().toString();
        return name.substring(0, name.length() - ".java".length());
    }

    /**
     * Loads a main class without initialising it.
     *
     * @param name its binary name
     * @return the class
     */
    private static Class<?> load(String name) {
        try {
            return Class.forName(name, false, CallerSurfaceTest.class.getClassLoader());
        } catch (ClassNotFoundException ex) {
            throw new IllegalStateException("A main source declares '" + name + "', which the test classpath lacks", ex);
        }
    }

    /**
     * Answers whether a source is a package's charter rather than a type.
     *
     * @param source the source
     * @return {@code true} for a {@code package-info.java}
     */
    private static boolean isCharter(Source source) {
        return source.path().getFileName().toString().equals("package-info.java");
    }

    /**
     * Answers whether a charter exists and reads as one: a doc comment ahead of the package line, and no
     * import.
     *
     * @param path where the package's {@code package-info.java} should be
     * @return {@code true} when it is there and both hold
     */
    private static boolean isChartered(Path path) {
        if (!Files.isRegularFile(path)) return false;
        String text = read(path);
        Matcher pkg = PACKAGE.matcher(text);
        if (!pkg.find()) return false;
        String head = text.substring(0, pkg.start());
        return head.contains("/**") && head.contains("*/") && !IMPORT.matcher(NOISE.matcher(text).replaceAll(" ")).find();
    }

    /** Every package a main source declares, read once. */
    private static Set<String> declaredPackages;

    /**
     * Returns every package a main source declares.
     *
     * @return the declared packages
     */
    private static synchronized Set<String> declaredPackages() {
        if (declaredPackages == null) {
            Set<String> declared = new TreeSet<>();
            sources().forEach(source -> declared.add(source.pkg()));
            declaredPackages = declared;
        }
        return declaredPackages;
    }

    /** Every main source, read once. */
    private static List<Source> sources;

    /**
     * Returns every Java source under the main root.
     *
     * @return the sources, in walk order
     */
    private static synchronized List<Source> sources() {
        if (sources == null) {
            if (!Files.isDirectory(MAIN))
                throw new IllegalStateException("No main source root at '" + MAIN.toAbsolutePath() + "' - run from the project directory");
            List<Source> read = new ArrayList<>();
            try (Stream<Path> files = Files.walk(MAIN)) {
                for (Path file : files.filter(path -> path.toString().endsWith(".java")).sorted().toList()) {
                    String text = read(file);
                    Matcher pkg = PACKAGE.matcher(text);
                    if (!pkg.find()) continue;
                    String stripped = NOISE.matcher(text).replaceAll(" ");
                    read.add(new Source(file, pkg.group(1), stripped, IMPORT.matcher(stripped).replaceAll(" ")));
                }
            } catch (IOException ex) {
                throw new UncheckedIOException(ex);
            }
            sources = List.copyOf(read);
        }
        return sources;
    }

    /**
     * Reads a source file.
     *
     * @param path the file
     * @return its text
     */
    private static String read(Path path) {
        try {
            return Files.readString(path, StandardCharsets.UTF_8);
        } catch (IOException ex) {
            throw new UncheckedIOException(ex);
        }
    }

}
