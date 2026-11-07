#pragma once
// 极简 JSON 解析/序列化：支持 object/array/string/number/bool/null，
// 满足 NeutronHP 的 DAG / schedule / 结果 schema（UTF-8 直通）。
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace nhp {
namespace json {

class Value;
using Members = std::vector<std::pair<std::string, Value>>;

class Value {
public:
    enum Type { Null, Bool, Number, String, Array, Object };
    Type type = Null;
    bool boolean = false;
    double number = 0.0;
    std::string string;
    std::vector<Value> array;
    Members object;

    bool IsNull() const { return type == Null; }
    bool IsObject() const { return type == Object; }
    bool IsArray() const { return type == Array; }
    bool IsNumber() const { return type == Number; }
    bool IsString() const { return type == String; }

    bool Has(const std::string &key) const
    {
        for (const auto &kv : object) {
            if (kv.first == key) return true;
        }
        return false;
    }

    const Value &At(const std::string &key) const
    {
        for (const auto &kv : object) {
            if (kv.first == key) return kv.second;
        }
        throw std::runtime_error("json: missing key " + key);
    }

    const Value &At(size_t index) const
    {
        if (index >= array.size()) throw std::runtime_error("json: index out of range");
        return array[index];
    }

    size_t Size() const { return type == Array ? array.size() : object.size(); }

    double Num(const std::string &key, double def = 0.0) const
    {
        return Has(key) ? At(key).number : def;
    }

    std::string Str(const std::string &key, const std::string &def = "") const
    {
        return Has(key) ? At(key).string : def;
    }

    std::string Dump() const;

    static Value MakeNull() { return Value(); }
    static Value MakeBool(bool v) { Value x; x.type = Bool; x.boolean = v; return x; }
    static Value MakeNumber(double v) { Value x; x.type = Number; x.number = v; return x; }
    static Value MakeString(const std::string &v) { Value x; x.type = String; x.string = v; return x; }
    static Value MakeArray() { Value x; x.type = Array; return x; }
    static Value MakeObject() { Value x; x.type = Object; return x; }

    void Set(const std::string &key, Value v)
    {
        type = Object;
        for (auto &kv : object) {
            if (kv.first == key) { kv.second = std::move(v); return; }
        }
        object.emplace_back(key, std::move(v));
    }
    void Push(Value v)
    {
        type = Array;
        array.push_back(std::move(v));
    }
};

namespace detail {

inline void SkipWs(const std::string &s, size_t &i)
{
    while (i < s.size() && (s[i] == ' ' || s[i] == '\t' || s[i] == '\n' || s[i] == '\r')) ++i;
}

inline std::string ParseStringRaw(const std::string &s, size_t &i)
{
    if (s[i] != '"') throw std::runtime_error("json: expected string");
    ++i;
    std::string out;
    while (i < s.size() && s[i] != '"') {
        if (s[i] == '\\') {
            ++i;
            if (i >= s.size()) break;
            switch (s[i]) {
                case 'n': out += '\n'; break;
                case 't': out += '\t'; break;
                case 'r': out += '\r'; break;
                case 'b': out += '\b'; break;
                case 'f': out += '\f'; break;
                case 'u': {
                    if (i + 4 >= s.size()) throw std::runtime_error("json: bad \\u");
                    unsigned code = static_cast<unsigned>(std::strtoul(s.substr(i + 1, 4).c_str(), nullptr, 16));
                    i += 4;
                    if (code < 0x80) {
                        out += static_cast<char>(code);
                    } else if (code < 0x800) {
                        out += static_cast<char>(0xC0 | (code >> 6));
                        out += static_cast<char>(0x80 | (code & 0x3F));
                    } else {
                        out += static_cast<char>(0xE0 | (code >> 12));
                        out += static_cast<char>(0x80 | ((code >> 6) & 0x3F));
                        out += static_cast<char>(0x80 | (code & 0x3F));
                    }
                    break;
                }
                default: out += s[i]; break;
            }
            ++i;
        } else {
            out += s[i++];
        }
    }
    if (i >= s.size()) throw std::runtime_error("json: unterminated string");
    ++i;
    return out;
}

Value ParseValue(const std::string &s, size_t &i);

inline Value ParseObject(const std::string &s, size_t &i)
{
    Value v = Value::MakeObject();
    ++i;
    SkipWs(s, i);
    if (i < s.size() && s[i] == '}') { ++i; return v; }
    while (true) {
        SkipWs(s, i);
        std::string key = ParseStringRaw(s, i);
        SkipWs(s, i);
        if (i >= s.size() || s[i] != ':') throw std::runtime_error("json: expected ':'");
        ++i;
        v.Set(key, ParseValue(s, i));
        SkipWs(s, i);
        if (i < s.size() && s[i] == ',') { ++i; continue; }
        if (i < s.size() && s[i] == '}') { ++i; return v; }
        throw std::runtime_error("json: expected ',' or '}'");
    }
}

inline Value ParseArray(const std::string &s, size_t &i)
{
    Value v = Value::MakeArray();
    ++i;
    SkipWs(s, i);
    if (i < s.size() && s[i] == ']') { ++i; return v; }
    while (true) {
        v.Push(ParseValue(s, i));
        SkipWs(s, i);
        if (i < s.size() && s[i] == ',') { ++i; continue; }
        if (i < s.size() && s[i] == ']') { ++i; return v; }
        throw std::runtime_error("json: expected ',' or ']'");
    }
}

inline Value ParseValue(const std::string &s, size_t &i)
{
    SkipWs(s, i);
    if (i >= s.size()) throw std::runtime_error("json: unexpected end");
    const char c = s[i];
    if (c == '{') return ParseObject(s, i);
    if (c == '[') return ParseArray(s, i);
    if (c == '"') return Value::MakeString(ParseStringRaw(s, i));
    if (c == 't') {
        if (s.compare(i, 4, "true") != 0) throw std::runtime_error("json: bad literal");
        i += 4;
        return Value::MakeBool(true);
    }
    if (c == 'f') {
        if (s.compare(i, 5, "false") != 0) throw std::runtime_error("json: bad literal");
        i += 5;
        return Value::MakeBool(false);
    }
    if (c == 'n') {
        if (s.compare(i, 4, "null") != 0) throw std::runtime_error("json: bad literal");
        i += 4;
        return Value::MakeNull();
    }
    size_t start = i;
    if (s[i] == '-' || s[i] == '+') ++i;
    while (i < s.size() && (std::isdigit(static_cast<unsigned char>(s[i])) || s[i] == '.' ||
                            s[i] == 'e' || s[i] == 'E' || s[i] == '-' || s[i] == '+')) {
        ++i;
    }
    if (i == start) throw std::runtime_error("json: bad value");
    return Value::MakeNumber(std::strtod(s.substr(start, i - start).c_str(), nullptr));
}

} // namespace detail

namespace detail {
inline void DumpValue(const Value &v, std::string &out)
{
    switch (v.type) {
        case Value::Null: out += "null"; break;
        case Value::Bool: out += v.boolean ? "true" : "false"; break;
        case Value::Number: {
            char buf[64];
            if (v.number == static_cast<int64_t>(v.number)) {
                std::snprintf(buf, sizeof(buf), "%lld", static_cast<long long>(v.number));
            } else {
                std::snprintf(buf, sizeof(buf), "%.6f", v.number);
            }
            out += buf;
            break;
        }
        case Value::String: {
            out += '"';
            for (char c : v.string) {
                if (c == '"' || c == '\\') { out += '\\'; out += c; }
                else if (c == '\n') out += "\\n";
                else out += c;
            }
            out += '"';
            break;
        }
        case Value::Array: {
            out += '[';
            for (size_t i = 0; i < v.array.size(); ++i) {
                if (i) out += ',';
                DumpValue(v.array[i], out);
            }
            out += ']';
            break;
        }
        case Value::Object: {
            out += '{';
            for (size_t i = 0; i < v.object.size(); ++i) {
                if (i) out += ',';
                Value key = Value::MakeString(v.object[i].first);
                DumpValue(key, out);
                out += ':';
                DumpValue(v.object[i].second, out);
            }
            out += '}';
            break;
        }
    }
}
} // namespace detail

inline std::string Value::Dump() const
{
    std::string out;
    detail::DumpValue(*this, out);
    return out;
}

inline Value Parse(const std::string &text)
{
    size_t i = 0;
    return detail::ParseValue(text, i);
}

inline Value ParseFile(const std::string &path)
{
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("json: cannot open " + path);
    std::ostringstream ss;
    ss << in.rdbuf();
    return Parse(ss.str());
}

inline void WriteFile(const std::string &path, const Value &v)
{
    std::ofstream out(path, std::ios::binary);
    if (!out) throw std::runtime_error("json: cannot write " + path);
    out << v.Dump();
}

} // namespace json
} // namespace nhp
